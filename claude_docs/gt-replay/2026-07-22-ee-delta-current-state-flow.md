# Delta GT-Replayer 現況 Flow 與程式碼實作細節

> 這份文件是「現況說明」，不是修改計畫。所有路徑相對於 repo 根目錄；行號來自
> `patrick/implement-ee-space` branch。修改計畫另見
> `claude_docs/gt-replay/2026-07-21-ee-delta-tracer-plan.md`。

---

## 0. 大圖：fake-hardware 上的三個 process

一個 fake-hardware delta replay 由三個 container/node 組成（`docker-compose.fake-hardware.yml`）：

| Service | Node | 角色 |
|---|---|---|
| `mock-robot` | `MockControllerNode` (`test/fake_hardware/fake_hardware_node.py`) | 假機器人：訂閱 `/commanded_ee_{arm}`，把收到的 pose **原封不動、瞬間**回吐到 `/ee_pose_{arm}`（帶 `sequence`） |
| `replay` | `DatasetGtReplayerNode` (`dataset_gt_replayer_node.py`) | GT 播放器：把 dataset 錄下的 action row 灌進 inference pipeline 的「模型預測」接縫，其餘全部沿用 base `LeRobotInferenceNode` |
| `gt-replay-verify` | `GtReplayVerifierNode` (`gt_replay_verifier_node.py`) | 裁判：比對 `replay` 發出的 `/commanded_ee` 與 dataset 的 `observation.state[t+1]`，寫 JSON report 判 PASS/FAIL |

閉環：`replay` 算出 absolute EE 指令 → 發到 `/commanded_ee_{arm}` → `mock-robot`
瞬間 echo 回 `/ee_pose_{arm}` → `replay` 讀到這個新 obs → 才敢往下走下一個 delta。
`gt-replay-verify` 在旁邊全程監聽兩個 topic 做數值比對。

---

## 1. 類別結構：DatasetGtReplayerNode 只覆寫幾個接縫

`DatasetGtReplayerNode(LeRobotInferenceNode)`（`dataset_gt_replayer_node.py:37`）。它只覆寫
base 暴露的幾個 hook，obs 讀取、EE 轉換、deque、`_publish_loop` 的 compose、發布、關機
全部沿用 base（不動）：

- `_setup_config`（`:76`）— 宣告 replay/homing 參數後再呼叫 `super()`。
- `_validate_required_params`（`:115`）— 需要 `dataset` 而非 `model_path`。
- `_load_run_metadata`（`:126`）— 從 `meta/info.json` + `conversion_config.yaml` 讀
  `action_type`，設 `model_type="gt_replay"`。
- `_setup_model`（`:173`）— 沒有模型；改成載入該 episode 的錄製 action rows。
- **`_produce_action`（`:343`）— 最關鍵的注入接縫。**

State：`self._gt_actions`（所有 row）、`self._replay_cursor=0`、
deque `self._classic_action_deque`（base 建的，`inference_node.py:81`，`maxlen=10`，裝 **delta**）。

兩個 timer 由 base 建（`inference_node.py:175-184`），都是 `1.0/control_freq`（預設 30Hz）：
- `_obs_timer` → `_obs_update`（讀 obs、算/注入 action、丟進 deque）
- `_publish_timer` → `_publish_loop`（從 deque 取 delta、compose、發布）
- 兩者在不同的 `MutuallyExclusiveCallbackGroup`，靠 `MultiThreadedExecutor` 併行，跨執行緒共用
  `_obs_lock`。

---

## 2. Homing 階段（絕對 pose，ramp 限速）

Homing 用的是**絕對 pose 比對**，但每 tick 的指令是 **ramp 限速逼近**（不是 delta）：

- `_setup_homing`（`:197`）：目標 = 該 episode **frame-0 的觀測 pose**（quat layout），
  `self._home_target_quat`。若非 EE 模式或 `home_before_replay=false` 就直接跳過。
- `_check_homing_arrival`（`:227`，由 `_obs_update` 每 tick 呼叫）：逐臂用
  `ee_runtime.pose_arrival_error()` 比 live obs vs 目標，取各臂最大值；當
  `max_pos_err<=home_atol_pos_m` 且 `max_rot_err<=home_atol_rot_deg` 就設
  `_homing_confirmed=True`。超過 `homing_timeout_sec` 則寫 `homing_failed` signal 並關機。
- `_publish_home_target`（`:288`，由 `_publish_loop` 每 tick 呼叫）：用
  `ee_runtime.ramp_toward_pose(current, target, home_max_pos_delta_m, home_max_rot_delta_deg)`
  每 tick 最多移動這麼多，再轉 rot6d 用 `_publish_ee_action` 發出。

參數預設（launch）：`home_atol_pos_m=0.05`、`home_atol_rot_deg=10.0`、
`homing_timeout_sec=30.0`、`home_max_pos_delta_m=0.01`、`home_max_rot_delta_deg=2.0`。

**Homing→Replay 轉換**：完全由 `_homing_confirmed` 翻成 `True` 驅動。在那之前，
`_obs_update` 呼叫完 `_check_homing_arrival()` 就 `return`（不會呼叫 `_produce_action`）、
`_publish_loop` 呼叫完 `_publish_home_target()` 就 `return`（不會 pop deque）。
→ **重點：homing 與 dataset 是 ee_abs 還是 ee_delta 完全無關，永遠是絕對 pose ramp。**

---

## 3. Action 注入接縫：`_produce_action`

Base 在「模型該產生預測」的位置呼叫 `action = self._produce_action(...)`
（`inference_node.py:839`），非 `None` 就 `append` 進 deque（`:841`）。覆寫版
（`dataset_gt_replayer_node.py:343-387`）機制：

1. **背壓**：`len(deque) >= maxlen-1` 就 return `None`（這 tick 不產生，避免 row 被 maxlen 擠掉）。
2. **結束判定**：cursor 溢位時依 `loop`/`hold_last` 處理（見 §6）。
3. 取當前 row：`action = self._gt_actions[self._replay_cursor].astype(np.float32)`（原生 on-disk
   編碼，delta dataset 就是 delta），直接當「模型輸出」回傳。
4. 正常路徑：`self._replay_cursor += 1`、`record_inference()`、`return action`。

**Cursor 前進位置（關鍵）**：`_replay_cursor += 1` 只在 `_produce_action` 裡，也就是在
**obs timer** 上、且受 deque 背壓節流 —— 不是在 publish timer 上。publish loop 的 hold-gate
會拖慢 deque 排空速度，透過背壓「間接」節流 cursor，但 cursor 自增這行本身只在 obs 路徑。

---

## 4. `_publish_loop` 的 ee_delta 分支（`inference_node.py:1021-1146`）

### 4a. 取得 obs 快照與 delta
- 進 branch 先在 `_obs_lock` 下快照 `_latest_obs = _ee_delta_latest_obs_quat`、
  `_latest_seq = _ee_delta_latest_obs_seq`（`:1022-1024`）。若 `_latest_obs is None`
  直接 return（不丟 delta）。
- delta 從 deque 頭取，但 **popleft 只在兩道 hold-gate 都通過之後**（`:1111`）。

### 4b. Compose 數學
`action = ee_delta_restore_step(_delta_popped, _latest_obs)`（`:1113`）。
`ee_runtime.py:153-200` 逐臂：`abs_xyz = obs_xyz + delta_xyz`（world-frame 加法）、
`R_abs = R_delta @ R_state`（world-frame extrinsic）、`gripper = delta_gripper`（絕對）。
→ `absolute_target = obs_pose ∘ delta`，**每個 publish tick 都用最新的 obs 重新 compose**。

### 4c. HOLD-GATE #1（sequence-based，`:1042-1076`）— fake-hardware only
比 `_ee_delta_last_published_seq` vs `_latest_seq`。當兩個 seq tuple 都存在、且**並非**每隻臂
的 latest seq 都嚴格大於上次發布的 seq → **HOLD**（不發、不 pop，`return` at `:1076`）。
意義：確認機器人已經「吃到」我上一個指令（echo 落地、sequence 前進）後，才前進到下一個 row，
維持 1:1 的 obs/delta 配對。real hardware 因 `get_ee_obs_sequence_snapshot()` 回 `None`，此 gate
是結構性 no-op。

### 4d. HOLD-GATE #2（position-proximity，`:1087-1109`）— 本 session 新增，兩種硬體都作用
比 live obs vs `_ee_delta_last_commanded_quat`（上次真正發出的絕對目標），逐臂用
`pose_arrival_error`。任一臂 `pos_err > _ee_delta_anchor_atol_pos_m`（預設 0.025m）或
`rot_err > _ee_delta_anchor_atol_rot_deg`（預設 6.0deg）→ **HOLD**（`return` at `:1109`）。
第一 tick（`_prev_commanded is None`）跳過。這是 real hardware 唯一的保護；對瞬間 echo 的 mock
基本是 no-op。

### 4e. 兩道 gate 都過之後（`:1111-1146`，順序）
1. `_delta_popped = deque.popleft()`（`:1111`）— **queue/cursor 在這裡才前進**
2. `_ee_delta_last_published_seq = _latest_seq`（`:1112`）— 更新 gate#1 state
3. compose（`:1113`）
4. 由 `action` 用 `ee_poses_from_chunk` 建 `_commanded_quat`（`:1117-1125`）
5. debug PUBLISHED log（`:1126-1145`）
6. `_ee_delta_last_commanded_quat = _commanded_quat`（`:1146`）— 更新 gate#2 state
→ 最後 `_publish_action(action)` 發出（`:1150-1155`）。

### 4f. HOLD 時的行為（已確認）
HOLD 時 delta **不 pop**（留在 deque 頭）、兩個 last_* state 都不更新、什麼都不發。下一 tick
重讀更新後的 `_latest_obs`、重評兩道 gate，通過就把**同一個** delta 對著更新後的 obs 重新
compose。→ compose 與 publish/pop 綁死，只在通過的 tick 發生。

### 4g. 目前既有的 DEBUG log（都 gated on `self._debug`，單行格式）
- `[ee_delta]{_progress} HELD(seq): {_latest_seq} not advanced past {_prev_seq}`（`:1072`）
- `[ee_delta]{_progress} HELD(pos): arm={_i} pos_err=...m rot_err=...deg (tol ...)`（`:1103`）
- `[ee_delta]{_progress} PUBLISHED{_tracking} obs=[...] delta=[...]`（`:1141`）
  其中 `_progress = " row={cursor}/{total}"`（僅 replayer 有），`_tracking` 是上一 tick 的追蹤誤差。

---

## 5. Mock 的 echo 與 sequence 機制（閉環的關鍵）

`MockControllerNode._setup_ee_mode`（`fake_hardware_node.py:233-290`）：發
`MockEEPose` 到 `/ee_pose_{arm}`、訂 `CommandedEEPose` 從 `/commanded_ee_{arm}`，有固定
100Hz 的 `ee_pose_timer` 重播目前 `_ee_state`。

- **Echo 是瞬間且精確、非 ramp**：`_ee_command_callback`（`:323-351`）把收到的
  pos/quat/gripper 原地複製進 `_ee_state[arm]`（docstring 明說 "perfect, instantaneous echo"）。
- **Sequence**：per-arm `_ee_seq_by_arm` 初始 0；**每收到一個指令 +1**（`:349`，一次 pose UPDATE
  一次），**不是每 publish tick +1**。publish tick 只是把目前 seq 值蓋上去（`:320`）。
  → 所以重播「尚未更新」的 pose 會重複同一個 seq 值，讓 consumer 分得出「這是舊 pose 的重播」。

Consumer 端（`strategies/multi_process.py`）：`mock_ee_pose_echo=true`（compose 對 replay 服務
硬寫 true）時才建 `SequenceStalenessGuard`（`:132-141`），並用 `_make_mock_ee_cb`（收 `MockEEPose`）；
`check(name, msg.sequence)` 判定 stale 則保留舊 anchor 不覆寫（`:258-302`）。
`get_ee_obs_sequence_snapshot()`（`:340-363`）回傳各臂最後接受的 seq；real hardware 回 `None`。

訊息型別：`CommandedEEPose`（header/pose/gripper，無 sequence，real+fake 共用）、
`MockEEPose`（= `CommandedEEPose base` + `uint64 sequence`，僅 fake）。

`SequenceStalenessGuard`（`ee_obs_sequence_guard.py`）`check()` 僅在 seq 嚴格大於該臂上次接受值
時 accept；有 warm-up grace（從未前進過的臂無條件 accept）與 `degraded_after_streak`（預設 50，
連續 stale 太久就 sticky 降級接受）。

---

## 6. Episode 結束

`_produce_action` 中 `_replay_cursor >= len(_gt_actions)` 時：
- `loop=True`：cursor 歸零重播。
- `loop=False`：首次觸發時 log + 寫 `{"status":"complete", rows_replayed}` signal。之後
  `hold_last=True` → 每 tick return `None`，最後指令持續 hold；`hold_last=False` → `rclpy.shutdown()`。

`_write_signal`（`:323`）只寫一次（`_signal_written` 護欄）；terminal status 有 `complete` /
`homing_failed` / `interrupted`（SIGTERM handler）。

---

## 7. Verifier 判定與已知殘留 flakiness

`GtReplayVerifierNode`（`gt_replay_verifier_node.py`）同時訂 `/commanded_ee_{arm}` 與
`/ee_pose_{arm}`。對第 n 個指令檢查 `published_cmd[n] == dataset.observation.state[n+1]`（轉 quat；
有 t+1 offset 與丟最後一 frame）。容差極嚴：`atol_pos_m=1e-4`、`atol_rot_deg=0.5`、
`atol_gripper_m=1e-4`。per-arm pass = 未 timeout 且無 fail 且 compared==expected 且 seed_confirmed；
overall = 各臂 AND。report 寫 `/workspace/reports/gt_replay_report.json`（host `REPORTS_DIR`）。

`gt_replay_correctness_test.py`：`ee-abs` 與 `ee-delta` 兩個 fixture；全部 `up -d`（detached，
所以**沒有任何 log 串到前景** → 這就是要做前景串流的原因）；`mock-robot`(healthy) →
`gt-replay-verify` → sleep 3s → `replay`；輪詢 report，讀 `report["all_passed"]` 判 PASS/FAIL。

**已知殘留 flakiness**（`claude_docs/gt-replay/2026-07-18-fake-hardware-architecture.md` §15）：
ee_delta 仍會間歇失敗，特徵是**兩臂在同一個 index 同時 fail**（`first_failures[0].index` 一致），
偏移量 ~5–60mm，遠超 1e-4m 容差。疑似是 `replay` endpoint 剛上線時的全域時序打嗝（DDS discovery），
而非 per-message race。**尚未 root-cause。**

---

## 下一步

修正計畫（前景串流 recipe + 每 tick 清楚編號 trace + 用 trace root-cause 殘留 flakiness）見
`claude_docs/gt-replay/2026-07-21-ee-delta-tracer-plan.md`，待使用者看完本文件後給出調整指示。
