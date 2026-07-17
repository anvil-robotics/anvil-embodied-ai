# EE Delta-Flow 管線 — 架構與設計參考文件

> 本文為 `claude_docs/ee-delta-architecture-report.md` 的繁體中文翻譯版本。程式碼識別字、
> 函式簽章、設定欄位、`action_type` 字串、檔案路徑與行號等技術性內容維持原文，不做翻譯。

**分支：** `patrick/implement-ee-space` · **產出日期：** 2026-07-16
**範圍：** 本分支開發的七個項目——術語改名、`ee_delta_forward`/`ee_delta_inverse`、
mcap_converter 轉換時烘焙、`EEDeltaTransform`/統計量、解耦的 delta-mode 發布迴圈、
假硬體 EE 擴充，以及 GT-replay 工具。

## 如何閱讀本文件

這是一份深度參考文件，不是進度報告。內容由淺入深組織：

- **第 1 層** — 每個功能一段：做什麼／為什麼／屬於哪個工作流程。
- **第 2 層** — 每個使用者面向的工作流程（資料轉換、訓練、推論、離線驗證）各一條
  執行追蹤，依呼叫順序列出每一個被碰到的 file:function。
- **第 3 層** — 依第 2 層所建立的順序，逐檔案、逐函式詳細說明，附上確切的目前行號、
  呼叫者／被呼叫者，以及不明顯的設計決策。
- **術語改名稽核**、**與計畫的偏差**、以及**已知臭蟲／粗糙之處**各自獨立成一節，
  放在文末，把散落在四個工作流程中的發現彙整起來。

每一份被引用的檔案都是從目前的工作樹（working tree）重新完整讀取的——這個工作樹
**同時包含**已提交到本分支的 commit，**以及**疊加在其上、尚未提交的變更（全篇使用
`git diff main`，而非 `git diff <merge-base>`，因為有相當大的一層第二次編輯是尚未
提交的）。行號皆對照目前狀態引用，而非計畫文件本身（現已過時）所引用的行號。
若計畫與程式碼有出入，兩個版本都會列出——偏差絕不會被悄悄抹平、當作沒發生過。

全篇引用兩份規劃文件：
- `claude_docs/ee-delta-flow-plan.md` — 設計計畫（旋轉數學、六項計畫、術語定案提案）。
- `claude_docs/ee-space-libero-vs-production-diagnosis.md` — 促成本分支工作的失敗
  診斷報告（以 n-0 逐 chunk 相對化 ＋ normalization 範圍壓縮作為主要成因）。

第三份文件 `docs/relative_ee_failure_analysis.md` 比前兩者都早——這是一份更早期、
後來已被取代的診斷（成因排序不同），本分支只對它動了一個小手術，加上兩行「歷史記錄」
的標頭註解；其實際結論並未與後來的診斷做調解（見第 3 層，anvil_eval 一節）。

---

# 第 1 層 — 功能總覽

**1. 術語改名（`ee_rel` → `ee_relative`）。** 既有的 n-0（chunk 起始錨點）機制被
端到端改名——`ee_rel_forward`/`ee_rel_inverse` → `ee_relative_forward`/
`ee_relative_inverse`、`EERelTransform` → `EERelativeTransform`、
`_delta_ref_state` → `_relative_anchor_state` 等等——原因是舊名稱「delta」與本分支
引入的新 Delta(n-(n-1)) 概念相衝突，而且一旦存在第二種非絕對表示法，「rel」也就不再
能泛指「任何非絕對的表示法」。這項改名純粹是為了讓兩種機制在程式碼、設定、日誌中
往後都維持不含糊，並且橫跨所有工作流程，因為 `action_type` 這個字串（並非直接，
而是作為相鄰概念）會流經資料轉換、訓練、推論與離線評估。

**2. `ee_delta_forward` / `ee_delta_inverse`。** `anvil_shared/ee_transform.py`
中一組新的 SE(3) 轉換函式，實作世界座標系、逐幀單步 delta：
`delta_xyz = action_xyz - state_xyz`（單純相減，不做旋轉）以及
`R_delta = R_action @ R_state.T`（世界／外部座標系組合），反向為
`abs = state + delta` / `R_abs = R_delta @ R_state`。這是全新撰寫的函式，而非
既有本體座標系 `ee_relative_*` 那對函式的包裝——因為目標組合方式（已對照 robosuite
1.4.0 的 OSC 控制器原始碼驗證過）用的是相反的座標系慣例與乘法順序。這是整個分支
其餘工作賴以建立的數學核心；它不屬於任何單一工作流程，但*被*資料轉換（烘焙）與推論
（還原）*使用*，並*由* GT-replay *驗證*。

**3. mcap_converter 轉換時烘焙。** 不再於訓練時即時計算 delta，`mcap_converter`
現在會在設定檔設有 `ee_action_encoding: "delta"`（`DataConfig` 新欄位，預設
`"absolute"`）時，把 `ee_delta_forward` 直接烘焙進硬碟上的 `action` 欄位。
`observation.state` 在兩種模式下都維持絕對值。這是一個靜態、可獨立檢視的值——
GT-replay 只需要檢查「硬碟上這個數字對不對」，而不必檢查「每次重新呼叫這段程式碼
是否都得到一樣的結果」。屬於**資料轉換**工作流程（`mcap-convert` CLI）。

**4. `EEDeltaTransform` / `_compute_ee_delta_stats`。** 訓練端的對應機制：一個
逐樣本的資料集 transform，把 `observation.state` 從四元數(8n) 轉換成 rot6d(10n)
版面（模仿 `EEAbsTransform` 只處理觀測的路徑，而非 `EERelTransform` 會做相對化的
路徑），同時完全不動 `action`——delta 在轉換時就已經烘焙好了。
`_compute_ee_delta_stats` 直接從靜態烘焙欄位讀出 mean/std/min/max 來計算
normalization 統計量（不做即時重播），並複製既有的 epsilon 下限與 rot6d-identity
夾限防護機制。屬於**訓練**工作流程（`anvil-trainer --action-type=ee_delta`）。

**5. 解耦的 delta-mode 發布迴圈。** 在 ROS2 推論端，`ee_delta` 模式的動作佇列現在
儲存的是模型輸出的原始 delta（而非事先還原好的絕對值）。一個與觀測取樣計時器各自
獨立的發布迴圈計時器，每個 tick 都讀取最新的真實觀測，並透過新的
`ee_delta_restore_step`（`ee_runtime.py`）即時計算
`absolute_target = obs_pose ∘ delta`——每一個 tick 都重新計算一次。這與 robosuite
每個執行步驟都重新錨定的做法一致（也就是 LIBERO 已驗證條件實際的做法），而不是
相對於同一個固定的 chunk 起始錨點做還原。屬於**推論**工作流程（ROS2
`inference_node.py` 啟動流程）。

**6. 假硬體 EE 擴充（第 2b 項）。** 既有只支援關節空間的模擬硬體節點
（`fake_hardware_node.py`）新增了一個 `CommandedEEPose` 發布者（模擬
`/ee_pose_<arm>`）以及一個訂閱 `/commanded_ee_<arm>` 的訂閱者，會把收到的指令
回聲回去，作為下一次發布的觀測值——也就是 `next_obs ≈ last_received_command`。
這是為了在多容器 CycloneDDS 環境下、不需要真實硬體，實際演練 delta-mode 發布迴圈
的自我修正回饋（`obs_pose ∘ delta`）所必須的。屬於**推論**工作流程，作為單元測試
與真實硬體之間的中間驗證層。

**7. GT-replay 工具（`anvil-gt-replay`）。** 一個全新、不依賴模型的 CLI
（`packages/anvil_eval/src/anvil_eval/gt_replay.py`），單獨驗證轉換數學本身：
對於 `--encoding absolute` 資料集，會對連續的真值姿態做
`ee_delta_forward`→`ee_delta_inverse` 的 round-trip；對於 `--encoding delta`
資料集，會從原始絕對姿態序列重新計算預期的烘焙 delta，並與 `mcap_converter`
實際寫入硬碟的內容比對。不涉及任何模型、任何 checkpoint。只要有任一 episode 超出
機器精度等級的容許誤差，就會以非零狀態退出。屬於**離線驗證**工作流程，是每一個
轉換出來的 delta 資料集在投入訓練運算資源之前，都必須先通過的第一道關卡。

---

# 第 2 層 — 每個工作流程的端到端流程

## 工作流程 A — 資料轉換（`mcap-convert`）

進入點：`packages/mcap_converter/src/mcap_converter/cli/convert.py:main`（第 584 行）。

1. `main()` 解析 argv（611-686）——**沒有任何 CLI 旗標對應 `ee_action_encoding`**；
   這個欄位只能透過設定檔指定。
2. 載入設定：`ConfigLoader.from_yaml(args.config)`（`loader.py:60`）→
   `load_yaml`（`loader.py:52`）→ `ConfigLoader.from_dict`（`loader.py:141-193`）。
   `from_dict` 讀取 `ee_action_encoding`（loader.py:150-156，預設 `"absolute"`，
   即時對照 `("absolute","delta")` 做驗證），並建構 `DataConfig(...)`。
3. `validate_config(config)`（`validators.py:44-127`）再次驗證
   `ee_action_encoding`（88-92），並在 `data_space != "ee"` 時拒絕
   `ee_action_encoding != "absolute"`（93-98）。
4. `main()` 第 710 行：`space_suffix = "ee-delta-space" if config.is_ee_delta
   else f"{config.data_space}-space"`——`is_ee_delta`（`schema.py:152-155`）
   是每一個消費端都檢查的唯一屬性；這是 delta 編碼唯一會改變**輸出路徑**的地方。
5. `LeRobotWriter.create_dataset(...)` → `_define_features`（`writer.py:180`）
   ——**不會**依 `is_ee_delta` 分支（只依 `is_ee`，writer.py:217）；絕對值與
   delta 兩種 EE 資料集會得到完全相同的 schema。
6. 針對每個 MCAP 檔案：`extractor.extract_frames(mcap_path, task)`
   （`extractor.py:647-895`）——真正做逐幀工作的產生器（generator）。它透過一個
   `prev_ee_state` 區域變數（第 748 行，episode 開始時為 `None`——每個 episode 都
   是真正的自我錨定，不會跨 episode 洩漏）貫穿 `_align_frame_at_cursor`
   （897-967，`prev_ee_state` 參數於第 906 行新增）傳入 `_align_ee_signals`。
7. `_align_ee_signals(ee_buffers, target_ts, prev_state=None)`
   （`extractor.py:1357-1424`）——**分支真正發生的地方**：先無條件建構絕對值的
   `state_abs`/`action_abs`（1390-1409），接著 `if self.config.is_ee_delta:`
   （1411）延遲匯入 `ee_delta_forward`（anvil_shared.ee_transform），設定
   `anchor = state_abs if prev_state is None else prev_state`（1416，第一幀
   自我錨定慣例），並計算 `action_out = ee_delta_forward(action_abs, anchor)`
   （1417）——對整個串接後的多手臂陣列呼叫一次，而不是逐手臂呼叫（逐手臂的迴圈是
   在 `ee_delta_forward` 內部透過 `n_arms_from_dims` 完成的）。`observation.state`
   原封不動地回傳（永遠是 `state_abs`）。
8. 控制流程從 `extract_frames` 往上回到 `convert.py` 的逐 episode 迴圈 →
   `writer.add_episode(...)` → `writer.finalize(dataset)`。

**`ee_action_encoding` 從頭到尾被讀取的位置：** YAML 欄位 → `loader.py:150-156`
→ `DataConfig.ee_action_encoding`（`schema.py:124`）→ `DataConfig.is_ee_delta`
屬性（`schema.py:152-155`）→ 剛好三個呼叫點：`convert.py:710`（輸出路徑）、
`extractor.py:1411`（分支到烘焙）、`validators.py:88-98`（輸入驗證）。沒有其他
檔案會讀取這個欄位。

**第 1 項（烘焙）在第 7 步接上。** 第 2 項（旋轉數學）是這個共用函式庫
`ee_delta_forward` 自身會呼叫的部分。術語改名與此工作流程無關（mcap_converter
自己的命名——`ee_action_encoding`／`action_type="ee_delta"`——一開始就是乾淨的，
從未使用過舊的「rel」／「delta」用語）。

## 工作流程 B — 訓練（`anvil-trainer --action-type=ee_delta`）

進入點：`anvil_trainer.train:main()`（`train.py:273-281`）→ `train()`
（`train.py:62-133`）。

1. `config = TrainingConfig.from_env_and_args()`（`train.py:75` →
   `config.py:186-476`）：`action_type = _pop_argv("action-type") or
   "joint_abs"`（204），對照 `_VALID_ACTION_TYPES = {"joint_abs","ee_abs",
   "ee_relative","ee_rel","ee_delta"}`（config.py:81, 220-224）做驗證，接著
   透過 `anvil_shared.action_types.normalize_action_type` 做正規化（226）——
   對 `ee_delta` 而言是無作用的，因為它不是任何東西的別名。
2. `config.validate_action_space()`（`train.py:82` → `config.py:515-586`）
   檢查 `self.action_type in ("ee_abs","ee_relative")`（543），以決定是否要求
   EE 資料集標記——**`ee_delta` 被排除在這個檢查之外，因此完全不會對它做任何
   資料集形狀驗證**（見「粗糙之處」）。
3. `with patched_lerobot(config):`（`train.py:105` → `patches.py:1181-1213`）
   建立 `TransformRunner(config)`（patches.py:144-165），登記了五個 transform
   實例（patches.py:146-153）——對 `ee_delta` 執行而言，只有
   `EEDeltaTransform.is_enabled(config)`（`transforms.py:303-304`，→
   `config.is_ee_delta`）回傳 `True`。
4. `runner.apply_metadata_patches()`（patches.py:1199 → 652-655）→
   `EEDeltaTransform.patch_metadata`（`transforms.py:332-337`）→
   `_patch_obs_state_shape_8n_to_10n`（transforms.py:174-209）對 lerobot 的
   `dataset_to_policy_features` 做 monkey-patch，讓 policy factory 看到的
   `observation.state` 是 10n 維（與 `EEDeltaTransform.apply` 實際輸出的維度
   一致）。
5. `runner.apply_dataset_patches()`（patches.py:1204 → 657-695）安裝
   `patched_getitem`（675-692），對每個樣本呼叫
   `EEDeltaTransform.apply(item, config)`（`transforms.py:306-330`）：透過
   `ee_obs_abs_forward` 轉換 `observation.state`（308, 316）；**完全不讀取或
   寫入 `item["action"]`**——delta 已經烘焙好了，因此整個「不會重複做兩次
   transform」的保證，完全仰賴這個方法根本不去碰這個鍵值（見「粗糙之處」，
   並沒有明確的防護機制，只是單純沒有寫這段程式碼而已）。
6. `runner.apply_val_loss_patch()`（patches.py:1205 → 697-846）安裝
   `patched_make_dataset`，每次執行只會（依 `_patched["done"]` 守衛）分派一次：
   `elif val_state.config.is_ee_delta: _patched_ee_stats =
   val_state._compute_ee_delta_stats(full_dataset, cfg)`（743-744）→
   **`TransformRunner._compute_ee_delta_stats`**（patches.py:451-559）：直接從
   `full_dataset.hf_dataset` 讀取 `actions_np`/`states_np`（沒有即時重播 transform），
   計算 epsilon 下限保護過的 mean/std/min/max（498-501），套用
   `_force_rot6d_identity`（503），透過 `ee_obs_abs_forward` 轉換觀測（518），
   並在被呼叫端把結果注入
   `train_dataset.meta.stats["action"]`/`["observation.state"]`
   （patches.py:826-828）之前，先記錄一行明確的
   `"[ee_delta_stats] COMPLETED and INJECTED (not the dataset-stats
   fallback)"` 日誌（542-548）——這才是 lerobot 的 normalizer 實際會讀取的
   資料集物件。
7. `runner.apply_checkpoint_patch()`（patches.py:1206 → 848-943）為每個
   checkpoint 寫入 `anvil_config.json`（`action_type`、`is_ee`、
   `is_ee_relative`、git 溯源資訊）——**`is_ee_delta` 並不在被持久化的欄位之列**
   （見「粗糙之處」）。
8. 在實際的訓練迴圈中，每一次 `__getitem__` 呼叫都會執行第 5 步的
   `EEDeltaTransform.apply`。

**第 4 項（EEDeltaTransform／統計量）就是整個工作流程。** 術語改名在此表現為
`EERelativeTransform`/`_compute_ee_relative_stats`/`is_ee_relative` 與新的
`EEDeltaTransform`/`_compute_ee_delta_stats`/`is_ee_delta` 在同一個
`TransformRunner` 裡以並列的兄弟關係存在。

## 工作流程 C — 推論（ROS2 啟動流程，`ee_delta` 模式）

進入點：`ros2/src/lerobot_control/launch/inference.launch.py` →
`inference_node.py:main()`（1370）→ `LeRobotInferenceNode.__init__`（56）。

**啟動：** `_setup_config()`（176）讀取 checkpoint metadata 並呼叫
`resolve_action_type(meta)`（`ee_runtime.py:54`，把舊有 `"ee_rel"` 對應到
`"ee_relative"` 的唯一關卡），設定 `self.action_type`/`is_ee`/`is_ee_relative`/
`is_ee_abs`/`is_ee_delta`（252-259）。`strategy.setup(...)`（65）→
`multi_process.py` 獨立地依 YAML 設定判斷 EE-vs-joint（
`ee_arms = {name: ac for ... if "ee_command_topic" in ac}`），並為 EE 手臂透過
`_setup_ee_subscriptions` 訂閱 `CommandedEEPose`（預設主題 `ee_obs_topic`，
`/ee_pose_<arm>`）——**這是與 `resolve_action_type` 各自獨立的第二個 EE 模式
偵測器，沒有任何機制強制兩者要一致**（見「粗糙之處」）。兩個各自獨立、在不同
callback group 上的計時器被建立：`_obs_timer → _obs_update` 與
`_publish_timer → _publish_loop`（133-142）——這個解耦計時器架構在本分支之前就
已存在；本分支的貢獻，是為 `ee_delta` 模式定義每個計時器*要做什麼*。

**每個 tick，`_obs_update()`（652），在 obs callback group 上：**
1. `strategy.get_observation(...)` → `multi_process.py:_build_observation()`
   ——EE 分支把每支手臂最新的 `CommandedEEPose` 訊息串接進
   `observation["observation.state"]`（四元數(8n)、絕對值——這是*量測*姿態）。
2. 因為 `self.is_ee_delta`（而非 `is_ee_relative`）：`ee_obs_abs_forward`
   把四元數(8n) 轉換成 rot6d(10n)，**絕對值、不做相對化**
   （inference_node.py:702-707）。
3. 原始四元數版面的觀測，被存入 `self._ee_delta_latest_obs_quat`
   （在 `self._obs_lock` 保護下，728-738）——這是跨執行緒交給 `_publish_loop`
   的傳遞點。
4. `_needs_restore = self.is_ee_relative` → 對 ee_delta 而言為 `False`（755）
   → ee_delta 永遠不會捕捉 `_relative_anchor_state`。
5. `model.select_action(observation)`（785）→ 原始 normalize 後的動作 →
   `postprocessor.process_action(action)`（804）做反 normalize——對 ee_delta
   而言，**這個值就是 delta 本身**，是物理單位，尚未與任何東西組合。
6. 這個原始 delta 直接被推入 `self._classic_action_deque`（838），
   **未經還原**。

**每個 tick，`_publish_loop()`（847），在各自獨立的 publish callback group 上：**
1. `if self.is_ee_delta:`（879）在 `self._obs_lock` 保護下讀取
   `self._ee_delta_latest_obs_quat`（880-881）——來自*最近一次* `_obs_update`
   tick 的最新觀測，通常與被彈出的那個 delta 是在不同的實際時刻計算出來的
   （這正是刻意設計的解耦）。
2. 若尚未有觀測值，就直接 `return`，**且不彈出**佇列中的 delta（883-886）
   ——不丟棄它。
3. `action = ee_delta_restore_step(self._classic_action_deque.popleft(),
   _latest_obs)`（887）→ `ee_runtime.py:147` → `ee_delta_inverse`：
   `abs_xyz = obs_xyz + delta_xyz`、`R_abs = R_delta @ R_state`
   （世界座標系——正是計畫「旋轉數學（已定案）」的公式）。
4. `_publish_action(action)` → `_publish_ee_action`（1038）→ 每支手臂：
   `ee_poses_from_chunk`（`ee_runtime.py:197`）→ rot6d→四元數姿態字典 →
   夾爪平移／縮放／夾限（1086-1089）→ 發布 `CommandedEEPose` 訊息到
   `/commanded_ee_<arm>`。

**對照——既有的 `ee_relative`（n-0）模式：** `_obs_update` 會把整個觀測窗口
相對於一個固定錨點做相對化，在每個新 chunk 開始時捕捉一次
`_relative_anchor_state`，並呼叫 `ee_relative_restore_chunk`
（`ee_runtime.py:103`，本體座標系）**對整個 chunk 只做一次**——已還原成絕對值的
動作進入同一個佇列；`_publish_loop` 只是彈出、沒有每個 tick 的組合運算。這就是
第 2 項引入的架構分歧：ee_delta 在*發布*時對照新鮮觀測做組合；ee_relative 在
*chunk 產生*時做（僅一次）組合。

**假硬體模擬迴圈（第 2b 項），獨立於上述之外：**
`fake_hardware_node.py:MockControllerNode._setup_ee_mode()`（187）建立一個
`CommandedEEPose` 發布者，發布到 `/ee_pose_<arm>`，以及一個訂閱
`/commanded_ee_<arm>` 的訂閱者 → `_ee_command_callback`（241）**把
`self._ee_state[arm]` 整個覆寫**成收到的指令（264-266，瞬間回聲、無任何模擬
動態）→ 一個各自獨立的計時器 `publish_ee_poses()`（221，預設 100 Hz，與
`control_freq` 各自獨立）把目前狀態重新發布為下一次觀測值。完整迴圈：
`inference_node._publish_loop` → `/commanded_ee_<arm>` →（DDS 傳輸）→
`fake_hardware_node._ee_command_callback` → 下一次 `publish_ee_poses()` tick
→ 真正的 `inference_node._obs_update` 透過 `multi_process.py` 的訂閱回呼
接收到——這就是讓 delta-mode 自我修正（`obs_pose ∘ delta`）在沒有真實硬體的
情況下，能被端到端演練的機制。

## 工作流程 D — 離線驗證

這個工作流程下住著兩個真正各自獨立的工具；請勿混為一談。

**D1. GT-replay（`anvil-gt-replay`，不依賴模型，「第一道關卡」）。** 進入點：
`gt_replay.py:main()`（239）→ `parse_args()`（215）。`_detect_encoding
(dataset_path)`（107，只在 `--encoding auto` 時呼叫）直接讀取
`<dataset_root>/conversion_config.yaml` 裡的 `ee_action_encoding` 欄位
（第二次、隨性的 YAML 讀取——**不是**透過 `mcap_converter` 自己的設定載入器）。
`main()` 直接讀取 `<dataset_root>/meta/info.json` 取得 `total_episodes`——
**完全不使用 `anvil_eval.dataset.EvaluationDataset`**，這是一個刻意偏離計畫的
決定（見「與計畫的偏差」）。每個 episode：`_load_episode_arrays(dataset_root,
episode_idx)`（125）直接透過 pandas 讀取 parquet 檔案——繞過
`LeRobotDataset`／訓練時的 transform。兩個重播函式：
- `_replay_episode_absolute`（155）：純數學自洽性檢查——對連續的真值姿態做
  `ee_delta_forward` 再 `ee_delta_inverse`，沒有外部參照（有紀錄在案的限制：
  這個模式無法偵測資料損壞，只能偵測數學本身壞掉——參見測試套件自己的 docstring）。
- `_replay_episode_delta`（171）：對照真實的轉換器輸出，檢查第一幀自我錨定
  不變量，接著重建 `action_abs = ee_obs_abs_forward(states)`，計算
  `expected = ee_delta_forward(action_abs[1:], states[:-1])`，並與**硬碟上
  實際烘焙的欄位**比對——這正是計畫第 3 項 GT-replay 涵義所要求的「硬碟上的
  數字對不對」檢查。

誤差對每個 episode 縮減為兩個純量（透過 `_max_pos_rot_err`/
`_rot6d_angle_diff_deg`，92/80，取所有手臂／所有幀的最大位置誤差(m)與最大旋轉
誤差(度)），對照機器精度等級的預設值（`atol_pos=1e-6`、`atol_rot_deg=1e-4`）——
在任何地方都沒有針對煙霧測試規模放寬，符合計畫「這道門檻不放寬」的要求。沒有
JSON/CSV 輸出——PASS/FAIL 只透過日誌輸出到 stdout，並以行程結束代碼發出信號。

**D2. 依賴模型的離線評估（`anvil-eval`，既有工作流程，已為 EE 擴充，但**沒有**
擴充給 `ee_delta`）。** 進入點：`anvil_eval/cli.py:main()`（91）。載入
`anvil_config.json` → `EvaluationDataset` → `EpisodeEvaluator.__init__`
（`evaluator.py:62`）正規化 `action_type` 並設定
`self.is_ee = action_type in ("ee_abs","ee_relative")`——**排除了
`ee_delta`**。`evaluate_episode`（93）只依 `is_ee_relative`/`is_ee_abs` 分支
處理觀測轉換與 chunk 還原——**完全沒有針對 `ee_delta` 的任何分支**，因此
ee_delta checkpoint 的觀測會維持 8 維四元數（錯誤——訓練好的 checkpoint 預期
的是 10 維 rot6d），也完全不會套用任何還原。回到 `cli.py`，`if
evaluator.is_ee:`（194）決定是否走 EE 專屬的指標空間轉換；對 `ee_delta` 而言
為 `False`，因此會落入 `metrics.py:compute_episode_metrics`（136，第 198 行的
判斷條件同樣排除了 `ee_delta`）中**通用、非 EE** 的指標路徑——產出一個把公尺、
無單位的 rot6d 分量、以及夾爪單位混在一起計算的無意義 MAE/MSE。`plotting.py`
的 `show_delta`/`_is_ee` 判斷條件（48、140、148）有著相同的疏漏，因此 ee_delta
執行結果完全不會有 EE 專屬的圖表。**這正是計畫第 5 項（「門檻降低」的離線評估）
在 `ee_delta` 上根本沒有實作**——見「與計畫的偏差」。

`anvil_eval_ros/cli.py`（另一個獨立的 ROS-in-the-loop MCAP 重播工具）也有
同樣的缺口：它的 `is_ee = action_type in ("ee_abs","ee_relative","ee_rel")`
檢查同樣排除了 `ee_delta`。

---

# 第 3 層 — 逐檔案、逐函式詳細說明

## A. 資料轉換管線

### `packages/mcap_converter/src/mcap_converter/cli/convert.py`
角色：CLI 進入點，統籌設定的載入／驗證、writer／extractor 建構、逐 episode
迴圈，以及最終報告。

- `main(args=None)`（584-…）：argparse（611-686，沒有
  `--ee-action-encoding` 旗標——只能透過設定檔指定，不像關節模式的
  `--act-from-obs` 有 CLI 覆寫選項——這是刻意的省略，並非疏漏：本檔案自己的
  設計原則是「delta 編碼影響更大，更適合釘死在有版本控管的設定檔裡」）。載入＋
  驗證設定（690-702）。計算 `space_suffix`／輸出目錄（704-717，delta 得到
  `ee-delta-space/`，第 710 行）。印出啟動橫幅（776-803），顯示
  `config.data_space`，但**從不顯示 `ee_action_encoding`**——delta 執行的
  主控台輸出，除了輸出路徑後綴之外，看起來與絕對值執行完全一樣。
- 轉換迴圈主體（約 200-500 行）：`LeRobotWriter` 建構（241-249）；EE 模式使用
  空的關節名稱字典（252-259，EE 有固定的逐手臂維度）；`conversion_config.yaml`
  重新序列化（303-349）——當沒有指定 `--config` 路徑時，**會遺失
  `ee_action_encoding`**（此欄位之所以在一般路徑下仍存在，只是因為原始 YAML
  被逐字複製）；EE 模式完全沒有 debug 圖表支援，不論絕對值或 delta（518-530，
  既有的缺口，本次未改動）。

### `packages/mcap_converter/src/mcap_converter/config/schema.py`
角色：統一（joint+EE）converter 設定的 dataclass。

- `DataConfig`（102-194）：`ee_action_encoding: str = "absolute"`（124），
  附有 9 行的設計理由註解（117-123）——是一個純量旗標，而不是新的
  `data_space` 值，因此 `is_ee`（148-150）與每一個既有的 EE 分支點都不受影響。
- `is_ee_delta`（152-155，新增）：`data_space == "ee" and
  ee_action_encoding == "delta"`——每個消費端都檢查的唯一屬性；除了
  loader／validator 自身在解析時的原始讀取之外，沒有其他地方直接檢查這個
  原始欄位值。

### `packages/mcap_converter/src/mcap_converter/config/loader.py`
- `from_dict(config_dict)`（141-193）：在委派給共用的 `validate_config` **之前**
  就先讀取＋驗證 `ee_action_encoding`（150-156）——這裡拋出的是普通的
  `ValueError`；`validators.py` 對同一個限制拋出的是 `ConfigurationError`。
  因為在實際呼叫鏈中 `from_dict` 一定先執行，`validators.py` 裡對應這個特定
  檢查的分支在實務上是死碼（只有在某個呼叫端直接建構 `DataConfig`、且不透過
  `ConfigLoader` 就呼叫 `validate_config` 時才會被觸發——某些單元測試確實刻意
  這麼做，以單獨測試 loader 這條路徑）。

### `packages/mcap_converter/src/mcap_converter/config/validators.py`
- `validate_config(config)`（44-127）：EE 分支（82-98）——EE 模式下
  `action_topics` 必須為空（83-87，既有檢查，與 delta 無關）；新增：
  `ee_action_encoding not in ("absolute","delta")` 為錯誤（88-92）；新增：
  在 `data_space != "ee"` 時，`ee_action_encoding != "absolute"` 為錯誤
  （93-98）——在 joint 設定上設定 delta 編碼會被明確拒絕，而不是悄悄忽略。

### `packages/mcap_converter/src/mcap_converter/core/extractor.py`
角色：真正做逐幀抽取／對齊的產生器——烘焙發生的地方。

- `extract_frames(mcap_path, task)`（647-895）：`prev_ee_state:
  Optional[np.ndarray] = None` 區域變數（748），每次呼叫都會重設（也就是每個
  episode 都是真正的自我錨定，因為這個函式對每個 MCAP／episode 只會呼叫一次，
  不會有跨 episode 的洩漏）。在**兩個**產出（yield）點都有更新——主要串流迴圈
  （809）與緩衝區清空尾端（850）——同一個函式有兩條產出路徑，都需要補上同樣的
  更新；很容易漏掉其中一個（本次沒有漏掉，但若這個函式未來再被修改，值得
  重新檢查）。
- `_align_frame_at_cursor(..., prev_ee_state=None)`（897-967）：純粹的管線
  傳遞，當 `config.is_ee` 時把 `prev_ee_state` 傳給 `_align_ee_signals`
  （954）。
- `_align_ee_signals(ee_buffers, target_ts, prev_state=None)`
  （1357-1424）——**核心烘焙函式**。逐手臂迴圈（1392-1406）建構
  `state_slices`/`action_slices`，一律絕對值——delta 轉換是在串接*之後*才做
  的（1408-1409），並非在逐手臂迴圈內部（與計畫描述在結構上有一處偏差——見
  「與計畫的偏差」）。`if self.config.is_ee_delta:`（1411）延遲匯入
  `ee_delta_forward`（比照本檔案既有對 `anvil_shared.rotation` 的延遲匯入慣例，
  第 1388 行，讓 mcap_converter 除非真的用到，否則不必對 `anvil_shared` 的
  轉換模組有硬相依）；`anchor = state_abs if prev_state is None else
  prev_state`（1416，第一幀自我錨定，與計畫及既有的 `ee_delta_forward`
  identity-delta 單元測試先例完全一致）；`action_out =
  ee_delta_forward(action_abs, anchor)`（1417，對整個串接後的多手臂陣列呼叫
  一次）。回傳的 `observation.state` 一律絕對值，`action` 依編碼分支。

### `packages/anvil_shared/src/anvil_shared/rotation.py`
角色：`ee_relative_*` 與 `ee_delta_*` 共用的底層 SO(3)/rot6d/四元數
primitives。並非本分支 delta 工作專屬新增（是被重用，而非新撰寫的），但其批次
（batch）版本（`quats_to_matrices`、`matrices_to_rot6d`、
`rot6ds_to_matrices`、`matrices_to_quats`）讓 `ee_delta_forward`/
`ee_delta_inverse` 能透過單純的 numpy 廣播（broadcasting），統一處理單一
逐手臂切片或 `(T, ...)` 批次，呼叫端完全不需要依形狀分支。
- **粗糙之處：** `rot6ds_to_matrices`（219-241，批次版本）對接近零範數的欄位
  會*夾限*到 `1e-10`，而不是拋出例外；非批次版本 `rot6d_to_matrix`
  （90-115）在同樣情況下則會*拋出* `ValueError`。批次版本的 docstring 把這個
  做法的理由寫成「為了避免掩蓋下游程式碼中的臭蟲」——這句話讀起來是反過來的，
  因為悄悄夾限通常正是*掩蓋*臭蟲的做法，而純量版本正是為了同樣理由才選擇拋出
  例外。這個不一致早於本分支就存在，但直接關係到稽核
  `ee_delta_forward`/`ee_delta_inverse` 在退化輸入下的數值行為，因為兩者都會
  呼叫批次版本的 primitives。

### `packages/anvil_shared/src/anvil_shared/ee_transform.py`
角色：`ee_relative_*`（改名）與 `ee_delta_*`（新增）這對 SE(3) transform，
加上只處理觀測的輔助函式與版面轉換器。

- `n_arms_from_dims(state_dim, action_dim)`（65-86）：未改動的驗證輔助函式，
  由 state 維度（必須是 8 的正整數倍）推導手臂數量，並交叉檢查 action 維度
  （必須是 `10 * n`）。每一個最上層的 transform 函式都會呼叫它來決定逐手臂
  迴圈的邊界。
- `ee_relative_forward`/`ee_relative_inverse`（89-228）：**由**
  `ee_rel_forward`/`ee_rel_inverse` **改名而來**——邏輯本身未改動。本體座標系：
  `R_state.T @ world_delta`（平移）、`R_state.T @ R_action`（旋轉），有一個
  `per_sample_state` 分支（130、200），會在單一參考 state 與逐時間步 state
  之間切換（用於計算資料集層級的統計量時，每一幀都需要自己的錨點）。這正是
  模組 docstring（28-34）現在明確標示為「已診斷出的真實硬體抖動失敗的根因」
  的機制——只為既有的 `ee_relative`／舊有 `ee_rel` action_type 保留。
- **`ee_delta_forward(action_abs, state)`（231-309，新增）：** 世界座標系
  正向轉換。平移（298）：`act_xyz - state_xyz`，刻意**不**乘上 state 的旋轉
  （與相對值那對函式不同）——docstring（241-244）說明這已對照 robosuite
  1.4.0 的 OSC 組合方式驗證過（`goal_position = current_position + delta`），
  不是隨意的選擇。旋轉（300-306）：`R_delta = R_action @ R_state.T`——與
  `ee_relative_forward` 的 `R_state.T @ R_action`**乘法順序相反**；若把這個
  順序寫反，會悄悄產生一個與本體座標系等價、卻掛著錯誤名稱的結果，而
  `test_ee_transform.py:712-735` 正是為了防範這件事而存在的測試（斷言
  `ee_delta_forward` 的輸出，在同一組非平凡輸入下，不等於
  `ee_relative_forward` 的輸出）。這裡不需要 `per_sample_state` 分支
  （270-272，與相對值那對函式不同）——這是真正的簡化，不只是行數變少：
  相對值那對函式之所以需要分支，是因為它的本體座標系平移項對 1 維 state 與
  批次 state 的計算方式不同，而世界座標系的平移項在兩種情況下都只是單純的
  逐元素相減。
- **`ee_delta_inverse(delta, state)`（312-380，新增）：** 精確的代數反函數。
  平移（370）：`state_xyz + delta_xyz`。旋轉（372-375）：`R_abs = R_delta @
  R_state`——與 robosuite 自身的 `goal_orientation = delta_rotation @
  current_orientation` 順序／方向一致。docstring（332-335）明確寫出代數證明
  （`(R_action @ R_state.T) @ R_state = R_action`，因為 `R_state` 是正交
  矩陣），而不是留給讀者自行重新推導。這正是解耦發布迴圈在推論時所呼叫的函式
  （透過 `ee_runtime.ee_delta_restore_step`）。
- `ee_obs_relative_forward`（383-443）：由 `ee_obs_rel_forward` 改名而來，
  邏輯未改動。
- `ee_obs_abs_forward`（446-487）：未改動的函式，但在 delta 路徑上變得
  新地重要——訓練端的 `EEDeltaTransform.apply` 與驗證端的
  `gt_replay.py`/`test_ee_encoding.py` 都用它，單獨從 `observation.state`
  重建 `action_abs`，因為不論編碼為何，觀測都維持絕對值。
- `ee_rot6d_to_quat_layout`、`ee_quat_layout_names`、`ee_action_to_poses`
  （490-599）：與 EE-delta 無關；是 `anvil_eval` CLI／ROS 發布路徑使用的
  版面輔助函式。

### `packages/anvil_shared/src/anvil_shared/action_types.py`（新檔案）
角色：原本打算作為舊有 `action_type` 別名的唯一權威來源。
- `ACTION_TYPE_ALIASES = {"ee_rel": "ee_relative"}`（25-27）。
- `VALID_ACTION_TYPES = frozenset({"joint_abs","ee_abs","ee_relative",
  "ee_rel"})`（30）——**不包含 `"ee_delta"`**，儘管本模組自己的 docstring
  宣稱它是 `anvil_trainer`/`anvil_eval`/`anvil_eval_ros`/ROS2 節點共用的
  來源。實務上 `anvil_trainer.config` 自己維護了一份**獨立**的
  `_VALID_ACTION_TYPES`（config.py:81），確實包含 `ee_delta`，且從未匯入
  這個 frozenset——因此本模組並非其 docstring 所宣稱的那個唯一權威來源
  （見文末「粗糙之處」彙整）。
- `normalize_action_type(action_type)`（33-42）：冪等（idempotent）的別名
  解析，是每一個新增的 action_type 都應該遵循的模式——`ee_delta` 不需要別名
  （沒有任何舊東西需要映射過來），這也是為什麼它從
  `VALID_ACTION_TYPES` 中缺席（這是一個不同層面的問題，與別名字典無關）
  才是真正的缺口所在。

### `packages/anvil_shared/src/anvil_shared/__init__.py`
- 把 `ACTION_TYPE_ALIASES`/`VALID_ACTION_TYPES`/`normalize_action_type` 與
  旋轉 primitives 匯入到套件命名空間（2-18），但**完全沒有從
  `anvil_shared.ee_transform` 匯入任何東西**，而 `__all__`（22-35）第 34 行
  仍列出 `"ee_obs_abs_forward"`。這是一個真實、可驗證的臭蟲：
  `from anvil_shared import ee_obs_abs_forward`（或 `import *`）會拋出
  `AttributeError`/`ImportError`。目前是潛伏、未被觸發的狀態，因為 repo 中
  每一個實際呼叫點都是直接 `from anvil_shared.ee_transform import
  ee_obs_abs_forward`（已透過全 repo 搜尋確認——沒有任何呼叫點使用套件根層級
  的形式）。

### 設定與測試固件
`configs/mcap_converter/openarm_ee_bimanual.yaml`、
`openarm_ee_bimanual_16x9.yaml`、`openarm_ee_left.yaml`（皆為 EE 設定，皆
因省略而預設 `ee_action_encoding` 為 `"absolute"`）、
`openarm_joint_bimanual.yaml`（無關——`openarm_bimanual_quest.yaml` 遷移到
統一設定格式的版本）、
`tests/smoke/fixtures/configs/mcap-converter-smoke-test-ee.yaml`、
`tests/smoke/fixtures/ee-session/*`（5-episode 固件集）、
`tests/smoke/fixtures/scripts/generate_ee_fixtures.py`。**這些檔案——以及
repo 中的任何其他 YAML——都從未設定 `ee_action_encoding: "delta"`**（見「與
計畫的偏差」，這是整個分支中最具體的一個「尚未接上」訊號）。

`tests/unit/mcap_converter/test_ee_encoding.py` 完全只透過在 Python 中直接
建構 `DataConfig(..., ee_action_encoding="delta")` 來演練 delta 路徑——從未
透過實際的 CLI／YAML 呼叫。

---

## B. 訓練管線

### `packages/anvil_trainer/src/anvil_trainer/train.py`
角色：薄的 CLI 進入點。
- `train(config)`（62-133）：用一個會自我還原的閉包包裝
  `lerobot_train.init_logging`（112-131），純粹是為了在同樣的格式下、緊接在
  「Output dir:」之前印出一行 resume 摘要日誌——只是外觀上的修補，與 ee_delta
  無關。
- `_ANVIL_HELP`（141-251）：**已完整記載 `ee_delta`**（161-164 範例指令、
  191-197 旗標說明）——這段說明文字是最新的，不像 `docs/training.md`
  （見「與計畫的偏差」）。

### `packages/anvil_trainer/src/anvil_trainer/config.py`
角色：`TrainingConfig` dataclass ＋ argv/env 解析 ＋ 驗證。
- `_VALID_ACTION_TYPES = {"joint_abs","ee_abs","ee_relative","ee_rel",
  "ee_delta"}`（81）——與 `anvil_shared.action_types.VALID_ACTION_TYPES`
  **另外獨立**的一組集合，後者缺少 `ee_delta`（見「粗糙之處」——兩個可能會
  逐漸失去同步的權威來源）。
- `is_ee`（170-172）：`action_type in ("ee_abs","ee_relative","ee_delta")`
  ——正確包含了 `ee_delta`。
- `is_ee_relative`（174-176）、`is_ee_abs`（178-180）、`is_ee_delta`
  （182-184，新增）：單純的相等比較。
- `from_env_and_args()`（186-476）：`data_space = "ee" if action_type in
  ("ee_abs","ee_relative") else "joint"`（343）——**排除了 `ee_delta`**，
  導致 ee_delta 執行的 checkpoint 被算進
  `model_zoo/joint-space/<dataset>/<run>/`，而不是 EE-space 目錄（真實的
  臭蟲，尚未修復，程式碼與計畫中都沒有標示出來）。
- `validate_action_space()`（515-586）：`self.action_type in ("ee_abs",
  "ee_relative")`（543）決定是否要求 EE 資料集標記——**排除了 `ee_delta`**，
  因此這個函式對 `ee_delta` 完全不做任何驗證（兩個分支都不會進入）。考量到
  計畫本身對其他 EE 類型的資料集／action-type 不匹配防護有多重視，這看起來
  像是疏漏。

### `packages/anvil_trainer/src/anvil_trainer/patches.py`
角色：所有的 lerobot monkey-patch，透過 `TransformRunner` 安裝／拆除。
- `_force_rot6d_identity(min_arr, max_arr, n_arms, dim_per_arm=10)`
  （61-77）：就地把 rot6d 維度（每支手臂的索引 3-8）夾限到 ±1——由所有三個
  `_compute_ee_*_stats` 方法共用；存在的原因是在 MIN_MAX normalization 下，
  把範圍強制設為精確的 `[-1,1]`，能讓 rot6d 不經 normalize 就原封不動通過
  （因為 rot6d 值本來就是單位向量分量，範圍已有界，並非真實資料剛好落在
  那個範圍）。
- `TransformRunner.__init__`（144-165）：登記 `EEDeltaTransform()`，與其他
  四個 transform 並列（151，本分支新增）。
- `_get_transform_details`（267-289）：
  `elif isinstance(transform, EEDeltaTransform): return "Delta(n-(n-1)):
  world-frame, baked on disk by mcap_converter — action untouched here"`
  （287-288）。
- `_compute_ee_relative_stats`（291-449）——既有 n-0 機制的統計量方法，由
  `_compute_ee_rel_stats` 改名而來；即時重播 `ee_relative_forward`，對
  `action_delta_indices` 的每一個 offset 都做一次，跨整個 horizon 彙整，並依
  episode 邊界遮罩。機制未改動，原封不動保留給 `ee_relative` 使用。
- **`_compute_ee_delta_stats(self, full_dataset, cfg)`（451-559，新增）：**
  這是 `ee_delta` **新增的**統計量方法。守衛：
  `if not self.config.is_ee_delta: return None`（474-475）；直接從
  `full_dataset.hf_dataset` 讀取 `actions_np`/`states_np`（484-486）——
  **沒有即時重播 transform**，因為 action 欄位已經是靜態值；
  `n_arms = n_arms_from_dims(...)`（491）；action 統計量透過單純的
  `mean()`/`std()`/`min()`/`max()`（498-501），並做 epsilon 下限保護
  （`np.where(std<1e-6, 1e-6, std)`，499——逐字複製了計畫要求的防護機制）；
  `_force_rot6d_identity(act_min, act_max, n_arms)`（503）；觀測統計量透過
  `ee_obs_abs_forward(states_np)`（518，與「模仿 EEAbsTransform 的觀測處理」
  這項決策一致）；明確的完成日誌（542-548）直接滿足計畫「必須證明這確實有跑、
  而非退回後備方案」的要求；`except Exception` 時退回，同樣有明確的
  `"[ee_delta_stats] FAILED"` 字樣（552-559）——`DataIntegrityError` 會被
  重新拋出，而不是被吞掉（550-551），因此只有真正非預期的失敗才會觸發悄悄
  退回後備方案。
- `apply_val_loss_patch`/`patched_make_dataset`（697-846）：分派鏈
  `is_ee_relative → _compute_ee_relative_stats；is_ee_abs →
  _compute_ee_abs_stats；is_ee_delta → _compute_ee_delta_stats；否則 None`
  （739-746）；統計量被注入
  `train_dataset.meta.stats["action"]`/`["observation.state"]`
  （825-828）——**這才是真正重要的那次寫入**，因為
  `_compute_ee_delta_stats` 內部自己對 `full_dataset.meta.stats` 做的就地
  修改（512、534）操作的是**另一個**資料集物件（`full_dataset` vs 經過
  篩選的 `train_dataset`），一旦 `full_dataset` 離開作用域，就是無效果的
  多餘操作（見「粗糙之處」）。
- `apply_checkpoint_patch`（848-943）：`anvil_cfg_base`（859-864）持久化
  `action_type`、`is_ee`、`is_ee_relative`——**沒有 `is_ee_delta` 這個
  鍵**，儘管這個屬性確實存在，且與 `is_ee_relative` 直接對應（見
  「粗糙之處」）。

### `packages/anvil_trainer/src/anvil_trainer/transforms.py`
- `EEAbsTransform`（217-266）：`EEDeltaTransform` 所模仿的既有模式——把觀測
  從四元數(8n) 轉換成 rot6d(10n)、action 直通；`_first_apply` 只守衛一次性的
  日誌記錄（已確認沒有數值上的差異，與計畫「已釐清、不是真正的風險」的結論
  一致）。
- **`EEDeltaTransform`（274-337，新增）：** `is_enabled`（303-304）→
  `config.is_ee_delta`。`apply(item, config)`（306-330）：守衛
  `"observation.state" not in item`（310-311）；透過
  `ee_obs_abs_forward` 轉換 `obs_np`（308, 316）；**完全不讀取或寫入
  `item["action"]`**（318-319，明確有寫註解）——這裡「避免雙重轉換」的保證
  是透過省略達成的，而不是明確的斷言；唯一能捕捉未來若不慎在此新增動作端
  邏輯的，是專屬的迴歸測試
  `test_ee_delta_transform.py::test_action_completely_unchanged_no_
  double_transform`。`patch_metadata`（332-337）委派給共用的
  `_patch_obs_state_shape_8n_to_10n`（174-209）。docstring（274-294）
  明確記載了「模仿 EEAbsTransform、而非 EERelativeTransform」這項刻意的
  設計決策。
- `EERelativeTransform`（345-425）：既有的 n-0 機制，由 `EERelTransform`
  改名而來；機制未改動。

### `packages/anvil_trainer/src/anvil_trainer/__init__.py`
- `EERelativeTransform` 在套件層級被匯出（22、35）；**`EEAbsTransform` 與
  `EEDeltaTransform` 都沒有**——只能透過
  `anvil_trainer.transforms.EEDeltaTransform` 存取，無法
  `from anvil_trainer import EEDeltaTransform`。對三個原本結構上完全對等的
  類別而言，這是不對稱的。

### 相鄰但不屬於 EE-delta 機制本身
- `packages/anvil_trainer/src/anvil_trainer/ema.py`（新檔案）：一個從零
  撰寫的 `EMAModel`（移植自 UMI 的 diffusion_policy EMA）加上
  `--no-ema`/`--ema-power` 等 CLI 旗標與 checkpoint 相關管線——這是對診斷
  報告中「失敗 checkpoint 是否有啟用 EMA？未能確定」這個未解問題的直接回應，
  往後為所有新 checkpoint 補上這個缺口。不論 `action_type` 為何都會統一
  套用；本分支雖然把它一併放進來，但與 delta 表示法的工作是彼此正交的。
- `tests/unit/anvil_trainer/test_umi_features.py`（新增，642 行）：上述
  EMA/DDPM-IP/DDIM-default 這一整包功能的測試，不是 ee_delta 機制本身。
- `scripts/training_metrics.sh`（新檔案）：一個通用的
  ACT/Diffusion/Pi0.5 訓練速度基準測試腳本，與 ee_delta 無關；其內部使用
  橫幅仍自稱為 `benchmark_training.sh`（一個改名遺留的不一致之處）。

---

## C. 推論管線（ROS2）

### `ros2/src/anvil_msgs/`（新套件）
`CommandedEEPose.msg`：`std_msgs/Header header; geometry_msgs/Pose pose;
float64 gripper`——同一個訊息型別同時用於外送指令（`/commanded_ee_<arm>`）與
內收觀測（`/ee_pose_<arm>`）；方向純粹是主題命名慣例。
`CMakeLists.txt`/`package.xml` 是標準的 `rosidl_generate_interfaces`
樣板程式碼。

### `ros2/src/lerobot_control/lerobot_control/ee_runtime.py`（新檔案，224 行）
角色：唯一一個為了 ROS 使用而包裝推論端 EE 數學 primitives 的地方——一層對
`anvil_shared.ee_transform` 的薄轉接層，刻意與 `inference_node.py` 分開，
讓這段數值邏輯不需要 `rclpy` 就能被匯入／測試。實質上取代了已刪除的
`delta_restore.py`，成為新的「執行期工具」模組，只不過是為了不同的功能
（EE 數學，而不是 `delta_restore.py` 原本承載的舊有關節空間 delta 機制）。

- `_ensure_anvil_shared()`（39）：延遲把 `packages/anvil_shared/src` 加入
  `sys.path`，在本檔案每一個公開函式的開頭才呼叫（而非在模組匯入時），讓
  匯入的開銷只有在真正用到時才需要支付。
- `resolve_action_type(cfg)`（54）：透過
  `anvil_shared.action_types.normalize_action_type` 正規化
  `cfg.get("action_type", "joint_abs")`——ROS2 端處理舊有 `"ee_rel"` 別名的
  唯一關卡。
- `read_checkpoint_anvil_config(model_path)`（69）：解析
  bare／`pretrained_model/`／HF 快取等 checkpoint 目錄結構，並讀取
  `anvil_config.json`——**重複**（其自身 docstring 第 72 行已承認：「比照
  `inference_node._read_checkpoint_metadata` 的路徑解析邏輯」）了
  `inference_node.py` 中同樣的邏輯，而非共用；只有 `inference_monitor_node.py`
  會呼叫它。
- `ee_relative_restore_chunk(chunk_np, obs_t)`（103）：`ee_relative_inverse`
  的薄包裝——**本體座標系**組合（`R_abs = R_state @
  rot6ds_to_matrices(delta_rot6d)`），既有 n-0 機制的數學未改動。接受 1 維或
  2 維的 `obs_t`（2 維時取最後一列）。
- **`ee_delta_restore_step(delta, obs_t)`（147，新增）：** docstring 明確
  與上方的 chunk 還原函式對照：「將單一 delta 與最新的觀測姿態組合……設計上
  每個發布 tick 都要呼叫一次」（154-159）。逐手臂：`abs_xyz = obs_xyz +
  delta_xyz`、`R_abs = R_delta @ R_state`——**世界座標系／外部座標系**，與
  chunk 還原函式相反的組合順序；與計畫的既定公式完全一致，一字不差。接受並
  回傳 1 維（單步）——正是 `_publish_loop` 每個 tick 呼叫點實際使用的形狀。
- `ee_poses_from_chunk(chunk_np, n_arms=None)`（197）：`ee_action_to_poses`
  的薄包裝，供**每一種** EE 模式（ee_abs/ee_relative/ee_delta）使用，因為
  呼叫這個函式時動作已經是絕對值。
- **設計缺口，未受任何強制約束：** 本模組的型別完全沒有區分「chunk」與
  「單步」——`ee_delta_restore_step` 理論上可以被傳入一個批次，並會把每一列
  都與同一個 `obs_t` 組合，若被誤用，會悄悄重現 n-0 式的過舊問題，違背了
  「每個 tick 都重新計算」的設計。這個慣例完全仰賴呼叫端自律。

### `ros2/src/lerobot_control/lerobot_control/strategies/multi_process.py`
角色：建構每個 tick 的 `observation` 字典；負責 EE-vs-joint 訂閱決策——
**對 EE 模式而言是關鍵檔案，不是樣板程式碼**（本文件研究過程中，這個檔案
原本被列為「只需確認、不必深入」，但實際上重要到值得完整說明）。
- `setup(...)`：`ee_arms = {name: ac for name, ac in arms_config.items()
  if "ee_command_topic" in ac}` 分支到新方法 `_setup_ee_subscriptions
  (ee_arms)`，或既有的 `_setup_joint_subscription`。**這是與
  `inference_node.py`（依 checkpoint 判斷）各自獨立的第二個 EE 模式偵測器**
  ——兩者必須依慣例保持一致（一份正確撰寫的 EE YAML，`ee_command_topic` 一定
  會搭配 EE checkpoint），但程式碼中沒有任何機制強制這件事；一旦不匹配，
  只會悄悄出錯，而不是明確報錯（`inference_ee.yaml` 的標頭註解有把這件事
  標示為已知陷阱，但只是文件層級的提醒，不是程式碼層級的防護）。
- `_setup_ee_subscriptions(ee_arms)`（新方法）：為每支手臂訂閱
  `CommandedEEPose`，QoS 設定為 RELIABLE/KEEP_LAST(10)；使用
  `_make_cb(name)` 閉包工廠，正確地在迴圈中對每支手臂綁定 `name`（避免經典的
  Python 延遲綁定閉包臭蟲）。
- `get_observation(...)`：EE 就緒檢查是
  `if not self._ee_state_by_arm: return None`——只檢查**至少一支**手臂曾
  發布過至少一次，而不是每支*已設定*的手臂都有。在雙臂設定中，若某一支手臂的
  發布者掛掉，`_build_observation` 會把那支手臂的欄位永遠悄悄填成
  `[0.0]*8`（預設值）——一個物理上毫無意義的零向量姿態——而不是把發布者
  掉線這件事表現出來。
- `_build_observation(images)`：EE 分支把
  `self._ee_state_by_arm.get(arm_name, [0.0]*8)` 逐手臂串接後直接回傳——
  與關節模式建構觀測的程式碼完全分開，沒有共用路徑。

### `ros2/src/lerobot_control/lerobot_control/inference_node.py`
角色：推論的統籌者。完整的每 tick 追蹤見第 2 層；以下是尚未在那裡涵蓋到的
函式：
- `_setup_config()`（176）：設定 `is_ee`/`is_ee_relative`/`is_ee_abs`/
  `is_ee_delta`（252-259）與 `ee_abs_uses_rot6d_obs`（263-265，一個以資料
  驅動的 `obs_state_dim % 10 == 0` 判斷法，用來區分舊有的四元數觀測
  `ee_abs` checkpoint 與新的 rot6d 觀測 checkpoint，而非用一個儲存的旗標）。
- `_read_checkpoint_metadata()`（291）：與
  `ee_runtime.read_checkpoint_anvil_config` 重複同樣的路徑解析邏輯，但也會
  讀取 `config.json`（影像形狀、模型類型）——無法輕易委派給那個共用輔助
  函式。
- `_obs_update()`/`_publish_loop()`（652/847）：見第 2 層。內部值得一提的
  設計決策：`_will_run_forward` 判斷法（740-748）會偷看模型內部動作佇列的
  長度，來判斷這個 tick 是否會真的呼叫模型——只用於延遲追蹤的紀錄，不影響
  控制流程——會查看兩種不同的 lerobot 內部屬性名稱（ACT 用
  `_action_queue`、Diffusion 用 `_queues["action"]`），因為 lerobot 的類別
  在這裡並未統一。
- `_publish_ee_action(action)`（1038）：由 ee_abs/ee_relative/ee_delta 共用
  （執行到這裡時，動作一律已經是絕對值）——建構 `CommandedEEPose`，套用
  夾爪平移／縮放／夾限（1086-1089）。
- `_publish_hold_position()`（1312）：對所有 EE 模式明確跳過
  （1321-1323）——anvil-workcell 控制器會自主維持最後一次的指令姿態；
  發送一個全零的 `Float64MultiArray` 會被誤判為關節指令。
- **確認很可能存在的臭蟲——`self._obs_lock`：** 只在
  `_setup_vla_inference()`（551）內部建立，而該函式只在
  `self._is_vla`（411-413）為真時才會被呼叫。但 `_obs_update`（737）與
  `_publish_loop`（880）在 `self.is_ee_delta` 為真時，都會無條件執行
  `with self._obs_lock:`——而依計畫，`ee_delta` 的目標架構是
  **Diffusion**，永遠不會是 `_is_vla`。這很可能會在任何一個經典（非 VLA）
  的 `ee_delta` 模型第一次 `_obs_update` tick 時，就拋出
  `AttributeError: 'LeRobotInferenceNode' object has no attribute
  '_obs_lock'`。**本次研究過程並未實際跑一個 ee_delta checkpoint 通過這個
  節點來驗證——標示為在信任這整套解耦發布迴圈是否真的能運作之前，優先順位
  最高的待確認事項。**

### `ros2/src/lerobot_control/lerobot_control/test/fake_hardware/
fake_hardware_node.py`（第 2b 項）
角色：整個 ROS2 主題介面（含新的 EE 部分）的獨立整合測試替身。單執行緒
（`rclpy.spin`，沒有 executor 併發），因此即使 `self._ee_state` 是在訂閱
回呼中被寫入、又在計時器回呼中被讀取，也完全不需要任何鎖，因為兩者都在同一個
執行緒上執行。
- `_setup_ee_mode()`（187）：為每支手臂用一個任意的起始姿態初始化
  `self._ee_state`（明確有寫註解：「不具物理意義；這是軟體時序／接線的煙霧
  測試，不是動力學模擬器」）；為每支手臂建立一個 `CommandedEEPose` 發布者
  （`/ee_pose_<arm>`）與一個訂閱者（`/commanded_ee_<arm>`）。
- `publish_ee_poses()`（221）：計時器回呼，頻率 `ee_pose_fps`（預設
  100 Hz，與真實節點的 `control_freq` 各自獨立）——原封不動地發布目前狀態。
- `_ee_command_callback(arm, msg)`（241）：驗證所有 8 個數值皆為有限值
  （否則拋出 `SystemExit(1)`，與關節模式相同的嚴格失效契約）；接著**整個
  覆寫** `self._ee_state[arm]`（264-266）——這是一次字面上的瞬間完美回聲，
  而不是加權混合或有物理積分過程的一步，模組 docstring 明確標示這是刻意
  排除在寫實模擬之外的範圍。
- 關節模式的程式碼未改動；「EE 模式下不會啟動關節主題／計時器」這項宣稱，
  是靠 `__init__` 中的 `if self._ee_mode: ... else: ...` 分支在結構上保證的，
  而非任何逐主題的執行期檢查。
- **值得明確重申：** 因為這個回聲是完整的狀態覆寫、沒有任何模擬延遲，這個
  模擬節點無法演練「如果真實手臂還沒追上最後一次指令怎麼辦」這類時序邊界
  情況——考量到真實硬體*確實*會有這種滯後（計畫本身第 6 項對「有界追蹤
  延遲」的擔憂），這是一個真實的限制。這是有文件記載的範圍界線，不是臭蟲——
  但一個在稽核「這個模擬節點是否真的能在寫實條件下驗證自我修正迴圈」的讀者，
  應該知道答案是「只在零延遲的極限情況下」。

### `ros2/src/lerobot_control/lerobot_control/inference_monitor_node.py`
- `__init__`（43）：透過 `read_checkpoint_anvil_config` →
  `resolve_action_type` 解析 `action_type`（自我偵測，即使啟動器沒有正確
  串接環境變數也不受影響）；或退回使用 ROS 參數。兩條路徑最終都會經過
  `resolve_action_type`，因此 CSV 中的 `# action_type:` 標頭一律是標準化
  後的名稱。**本檔案中完全沒有任何 `ee_delta` 專屬的處理**（不是崩潰風險，
  只是一個未經測試的缺口——見「與計畫的偏差」）。

### `ros2/src/lerobot_control/lerobot_control/eval_recorder_node.py`
- `self._is_ee = self._action_type in ("ee_abs","ee_relative","ee_rel")`
  ——**直接對照原始 ROS 參數字串**檢查一個 3 元組，而不是呼叫
  `resolve_action_type`——是 ROS2 樹中唯一一個重新自行實作別名檢查、而非
  使用共用關卡的地方。**這個 3 元組中沒有 `ee_delta`**——一個 ee_delta
  checkpoint 透過這個節點執行時，會悄悄落入關節模式分支，訂閱錯誤的主題
  （`Float64MultiArray`），什麼有意義的內容都記錄不到，卻也不會報錯。
- `_on_gt_ee`/`_on_pred_ee`（新增）：`CommandedEEPose` → 攤平的
  `[x,y,z,qx,qy,qz,qw,gripper]` 清單。
- `_compute_raw_ground_truth`（舊有的關節空間 delta GT 重建）**被整個
  刪除**——屬於下方描述的更大範圍移除工作的一部分。

### `ros2/src/lerobot_control/lerobot_control/action_limiter.py`
純關節空間的安全限制器（任何 EE 模式都會完全跳過它）。這裡的改動純粹是
移除舊有的 `delta_exclude_joints` 死區邏輯——本來就從未新增過任何 EE 專屬
邏輯，因為 EE 模式從一開始就完全繞過這個類別。

### 設定、腳本、文件（支援性質）
- `configs/lerobot_control/inference_ee.yaml`（新增）：在標頭註解中直接
  記載了兩種 EE 偵測機制；每支手臂的 `gripper_factor`/`gripper_min`/
  `gripper_max` 新增可調參數。
- `docker-compose.fake-hardware.yml`：新增 `EE_MODE`/`EE_ARMS`/
  `EE_POSE_FPS` 環境變數，串接到模擬控制器的 ROS 參數。
- `tests/smoke/fixtures/configs/inference-eval-smoke-test-ee.yaml`
  （新增）：明確把自己的範圍限定為演練「`ee_rel` 這個舊別名對應到
  `action_type=ee_relative`」——**這份固件中不存在任何 `ee_delta` 情境**
  （與 `eval_recorder_node.py` 的缺口一致：ee_delta 的 ROS 端發布迴圈，在
  這個測試套件中完全沒有煙霧測試涵蓋，只有數學本身的單元測試）。
- `docs/inference.md`：新增「EE 模式假硬體」一節，說明 `EE_MODE=true`
  的工作流程。

### 這個工作流程中超出計畫範圍、未被提及的變動
整個舊有關節空間 `delta_obs_t`/`delta_sequential` action-type 機制，在
ROS2 端被整個移除：`delta_restore.py` 被刪除、`eval_recorder_node.py` 的
`_compute_raw_ground_truth` 被刪除、`action_limiter.py` 的
`delta_exclude_joints` 被移除、`docker-compose.eval.yml` 的
`EVAL_USE_DELTA_ACTIONS`/`EVAL_DELTA_EXCLUDE_JOINTS` 環境變數被移除、
`plot_monitor_csv.py` 的舊 CSV 相容分支被移除。這是一次真實、可觀的功能
移除，計畫中完全沒有提到有做過這個決定——讀起來像是順手一併整理的工作
（計畫中「delta 命名衝突」這項擔憂，指的是 `_delta_ref_state`，但
`delta_obs_t`/`delta_sequential` 是第三個、看起來已經沒人在用的、完全獨立
的關節空間機制，在同一波清理中被一併清掉了）。

### 已確認與本分支 EE-delta 工作無關（各一行）
`image_worker.py`（全 episode JPEG 錄影功能）、`mcap_player_node.py`
（死碼刪除，與 EE 無關）、`docker/inference/Dockerfile`/`entrypoint.sh`
（錄影功能相關管線 ＋ DDS 環境變數整理）。

---

## D. 離線驗證

### `packages/anvil_eval/src/anvil_eval/gt_replay.py`（新增，尚未加入版本控管）
角色：獨立、不依賴模型的 CLI 關卡——計畫第 3 項，「第一道關卡、門檻嚴格」。
延遲匯入 `anvil_shared.ee_transform`/`anvil_shared.rotation`/`pandas`/
`yaml`/`json`，都是在函式內部才匯入，而不是在模組最上層（讓模組匯入本身
保持輕量，並避免在 CLI --help 時就對 pandas 有硬相依）。

- `_pos_slices`/`_rot_slices`（70、75）：從一個**寫死**的
  `_ACTION_DIM_PER_ARM = 10`（54）推導逐手臂 `(start,end)` 切片——沒有驗證
  `action_dim % 10 == 0`；形狀錯誤的資料集會因為整數向下取整而悄悄切錯。
- `_rot6d_angle_diff_deg(r6d_a, r6d_b)`（80）：rot6d→旋轉矩陣→
  `Rdiff = Ra @ Rb.T`→四元數→
  `angle = 2*arccos(clip(|qw|,0,1))`——使用 `|qw|`（絕對值）是刻意用來
  處理四元數雙重覆蓋問題（q 與 -q 代表同一個旋轉）的做法，與診斷報告 §1.3
  中的免疫論證一致。
- `_max_pos_rot_err`（92）：把一個 episode 縮減為**最大**（而非平均）的
  位置／旋轉誤差，橫跨所有手臂／所有幀——對一個嚴格的關卡而言恰如其分，
  即使整個 episode 平均下來沒問題，只要有一幀壞掉就必須算失敗。
- `_detect_encoding(dataset_path)`（107）：直接讀取
  `conversion_config.yaml`——第二次、隨性地讀取一個 `mcap_converter` 自己
  寫入的欄位，並未透過 mcap_converter 自己的設定 schema／loader。
- `_load_episode_arrays(dataset_root, episode_idx)`（125）：直接透過
  pandas 讀取 parquet，明確繞過 `LeRobotDataset`／訓練時的 transform
  （docstring 127-130：「讓這個工具獨立於任何訓練時的 transform」）——這是
  刻意的選擇，但它重複了 `EvaluationDataset` 已經為依賴模型的工作流程實作過
  的資料集載入邏輯，兩者之間沒有任何共用程式碼；其中一邊 episode 邊界邏輯的
  臭蟲修復，不會傳播到另一邊。
- `_replay_episode_absolute`（155）：anchor=`states[:-1]`，
  gt=`actions[1:]`；`delta = ee_delta_forward(gt, anchor)`；
  `recon = ee_delta_inverse(delta, anchor)`；誤差＝recon 與 gt 之比較。
  要求 `T>=2`。
- `_replay_episode_delta`（171）：對照真實的轉換器輸出，檢查第一幀自我
  錨定不變量（184-188，**唯一**一處對照真實轉換結果、而非只對照獨立的
  transform 函式，端到端測試這個慣例的地方）；當 `T>=2` 時，重建
  `action_abs = ee_obs_abs_forward(states)`，計算
  `expected = ee_delta_forward(action_abs[1:], states[:-1])`，並與硬碟上
  實際的 `actions[1:]` 比對——這是「烘焙欄位對不對」的檢查，而不是一次
  round-trip。
- `main()`（239）：統籌整個流程；
  `if n_total == 0: ... sys.exit(1)`（323-325）——一個所有 episode 都被
  跳過的資料集（全部少於 2 幀，或根本沒找到）算作硬性 FAIL，而不是悄悄
  無動作——這是刻意的嚴格關卡設計。
- **粗糙之處——單位不匹配：** 第一幀容許誤差（`main()` 第 296 行）使用
  `args.atol_rot_deg * 1e-2` 作為一個無單位 rot6d L2 誤差的門檻——把一個
  以「度」為單位的容許誤差乘上 `1e-2` 來得到一個無單位 L2 誤差的門檻，是
  數值上的權宜之計，而非有原則的推導（296-298 行的註解也承認了這一點）。
  如果改成獨立命名的常數，會清楚許多。
- **粗糙之處：** 完全沒有 JSON/CSV 輸出——PASS/FAIL 只透過 stdout 日誌加上
  行程結束代碼呈現，與 `anvil-eval` 自己會寫出結構化摘要的慣例不一致。

### `tests/unit/anvil_eval/test_gt_replay.py`（新增，尚未加入版本控管）
完全使用合成資料，沒有真實的 mcap／parquet 固件——因此這些測試驗證的是
「重播邏輯本身」的自洽性，而不是 `mcap_converter` 實際 `_align_ee_signals`
輸出的整合性臭蟲；沒有任何端到端、以固件為基礎的測試，橋接這兩者。
`TestReplayEpisodeAbsolute::
test_round_trip_exactness_is_anchor_content_independent`（76）在自己的
docstring 中明確記載「absolute」模式即使在一個被破壞的動作陣列上也會通過
（沒有外部參照）——「這是設計上如此，不是關卡的弱點」，把偵測資料損壞的
責任交給「delta」模式的 `test_corrupted_baked_column_fails`（124）。

### `packages/anvil_eval/src/anvil_eval/cli.py`
`main()`（91）：`if evaluator.is_ee:`（194、229）決定是否在呼叫
`compute_episode_metrics` 前先做 `ee_rot6d_to_quat_layout` 轉換——對
`ee_delta` 而言永遠不會進入，因為 `evaluator.is_ee` 從不包含它。

### `packages/anvil_eval/src/anvil_eval/evaluator.py`
- `EpisodeEvaluator.__init__`（62）：`self.action_type =
  normalize_action_type(...)`（84）；`is_ee`/`is_ee_relative`/
  `is_ee_abs` 旗標（85-87）驅動 `evaluate_episode` 中的每一個分支——
  **完全沒有 `is_ee_delta` 旗標**。
- `evaluate_episode`（93）：`_relative_anchor_state`（91，每個 episode 於
  第 113 行重設）——這是**改名後的既有 n-0 機制**，在這裡被重用於
  evaluator 自己的 chunk 還原邏輯，是與 mcap_converter 烘焙 delta 不同的
  另一件事；閱讀本檔案時請不要把兩者搞混。`_abs_shadow_queue`（115）
  刻意「影子式」複製模型自己的動作佇列，而不是就地修改它。`_is_new_chunk`
  偵測（142-146）仰賴 lerobot 私有的 `_queues` 屬性名稱——脆弱，沒有轉接層。
  觀測轉換在 `self.preprocessor` 執行*之前*就進行（160-181），原因是
  「資料集儲存的是 8 維四元數觀測；checkpoint 的 normalizer 統計量是
  10 維 rot6d（訓練時已修補過）」——這其實是在小規模地手動重新實作
  `EEAbsTransform`/`EERelativeTransform` 在訓練時所做的事；這正是一個
  `ee_delta`-aware 分支也會需要、但目前並不存在的觀測轉換邏輯。

### `packages/anvil_eval/src/anvil_eval/metrics.py`
- `EEMetrics`（12）：`position_pass`/`orientation_pass` 直接在類別中寫死
  `0.02` 公尺／`0.0873` 弧度的門檻——**同樣的**門檻又在
  `compute_summary_metrics`（282）與 `reporting.py` 的
  `_compute_aggregate`（82，用的是*第三種*單位慣例——`5.0` 度）中被獨立
  重新宣告——三份相同魔術數字的複本，目前彼此一致，但未來任何調整都有
  漂移風險。
- `compute_ee_metrics`（65）：預期的是 8 維四元數版面（不是
  `gt_replay.py` 所使用的 10 維 rot6d 動作版面）——這是 CLI 明確會先轉換
  進去的「指標空間」。方向誤差使用矩陣跡（trace）的測地線公式
  （`arccos((trace(rel)-1)/2)`）——與 `gt_replay.py` 以四元數為基礎的
  `2*arccos(|qw|)` 公式，數學上等價，但是各自獨立實作的。
- `compute_episode_metrics`（136）：`action_type in ("ee_abs",
  "ee_relative","ee_rel")` 這個判斷條件（198）**遺漏了 `ee_delta`**——這是
  第 5 項的缺口。當 EE 模式觸發時，所有通用純量指標會刻意被設為
  NaN／空字典（204-217），而不是計算出來卻不使用——原本的目的是避免產出
  一個混合單位、會誤導人的 MAE，但也因此讓 `ee_delta` 落入通用路徑這件事
  變得更糟、而非更輕微：它會產出一個「看起來像有效指標，其實不是」的數字。

### `packages/anvil_eval/src/anvil_eval/plotting.py`
- `plot_episode_joints`（20）：`show_delta = action_type in
  ("ee_relative","ee_rel") and ...`（48）——沒有 `ee_delta` 分支，因此
  ee_delta 執行結果不會有診斷用的底部區塊圖表，儘管類似的「烘焙 delta vs.
  delta 真值」比對，本來也會同樣有參考價值。
- `plot_monitor_signals`（115）：相同的判斷條件模式（140、148）；
  `_EE_DIM_NAMES`/`_EE_DIM_UNITS`（146-147）寫死了 10-維-每手臂版面——是
  `gt_replay.py` 的 `_ACTION_DIM_PER_ARM` 與 `anvil_eval_ros/cli.py` 的
  `_EE_DIMS_PER_ARM` 之外，第三個獨立寫死同一版面慣例的地方，三者都沒有
  共用 `anvil_shared` 裡的常數。

### `packages/anvil_eval/src/anvil_eval/reporting.py`
`_compute_aggregate`（53）：第三／第四次重新宣告 PASS/FAIL 門檻（82，
單位是度、不是弧度）；因為 `ee` 依照上述缺口，對 `ee_delta` 而言從未被
填入，這個函式會悄悄地在 ee_delta 執行結果的摘要 JSON 中，完全不產生
`"ee"` 這個鍵——下游看到的只是一份平淡的通用指標報告，沒有任何明顯錯誤，
若不仔細檢視輸出內容，這個缺口很容易被漏掉。

### `packages/anvil_eval_ros/src/anvil_eval_ros/cli.py`
本分支改寫，為 MCAP 重播評估的設定產生流程分出 EE vs joint 分支。
`_detect_arms_from_conversion_config`：EE 分支從 `observation_topics` 的鍵
讀取手臂名稱（EE 設定用的是這個，而不是 `action_topics`）。
`_EE_DIMS_PER_ARM = 10`（新的模組常數——這個版面慣例的第四個獨立寫死
之處）。`generate_inference_config`：`is_ee = action_type in ("ee_abs",
"ee_relative","ee_rel")`（原始未正規化）——**遺漏了 `ee_delta`**；一個
ee_delta checkpoint 在這裡會走 joint 分支，很可能會產生形狀錯誤的設定檔。
EE 的手臂順序後備值寫死為只有 `["right"]`——對於一個假設性的、沒有
conversion_config 的雙臂 EE checkpoint 情境，測試不足。`main()`：舊有的
`use_delta_actions`/`delta_exclude_joints` 環境變數介面
（`EVAL_USE_DELTA_ACTIONS`、`EVAL_DELTA_EXCLUDE_JOINTS`）被整個移除，
改成從 `anvil_config.json` 讀取單一的 `action_type` 字串——這是對
eval-ros 環境變數契約的一次真實、破壞性的簡化。

### `docs/relative_ee_failure_analysis.md`
這是**原始**的根因診斷報告（早於本分支的工作），結論是主要成因為觀測／
動作錨點不匹配（H1），世界／本體座標系混用是次要因素（H2）——這個排序
與後來、更嚴謹的 `claude_docs/ee-space-libero-vs-production-diagnosis.md`
**不同**，後者把 n-0 統計量彙整／normalization 範圍壓縮列為主因，並把
錨點不匹配降級為「(c) 不具可比性」。本分支只在標頭加上兩行說明，澄清這是
舊有 `ee_rel` 名稱下的歷史記錄——實際結論**並未**與較新的診斷做調解，因此
只讀這份文件的讀者，會得到一個過時的成因說法。

---

# 術語改名 — 彙整稽核

**整體狀態：基本上已完成，附有永久的舊有別名，僅有一處小小的不一致。**

已完成且透過全 repo 搜尋驗證（沒有任何殘留的舊名稱呼叫點）：
- `ee_rel_forward`/`ee_rel_inverse`/`ee_obs_rel_forward` →
  `ee_relative_forward`/`ee_relative_inverse`/`ee_obs_relative_forward`
  （`anvil_shared/ee_transform.py`）。
- `EERelTransform` → `EERelativeTransform`；`_compute_ee_rel_stats` →
  `_compute_ee_relative_stats`；`is_ee_rel` → `is_ee_relative`
  （`anvil_trainer`）。
- `_delta_ref_state` → `_relative_anchor_state`；`ee_rel_restore_chunk` →
  `ee_relative_restore_chunk`（ROS2 `inference_node.py`/`ee_runtime.py`）。
- 公開的 `action_type` 字串：`"ee_relative"` 是標準名稱；`"ee_rel"` 在每個
  地方都被接受作為**永久**別名，透過
  `anvil_shared.action_types.ACTION_TYPE_ALIASES`/`normalize_action_type`
  實現，並有測試載入舊有字串後斷言得到標準化結果——這滿足了計畫「必須永遠
  能讀取舊有 token」的要求，涵蓋現有的 31 個 checkpoint。
- `gt_replay.py` 與 `evaluator.py` 全程只使用新用語，且用法正確，因為
  `gt_replay.py` 是為新機制量身打造的，而 `evaluator.py` 已完整遷移。

唯一的不一致之處：`eval_recorder_node.py` 在行內重新實作了別名檢查
（`action_type in ("ee_abs","ee_relative","ee_rel")`），而不是呼叫
`resolve_action_type`/`normalize_action_type`——目前功能上是正確的，但這也
剛好是唯一一個同時遺漏了 `ee_delta` 的地方（見下方），這並非巧合：一個
單獨、非 DRY 的重新實作，正是這種疏漏會滲漏進去的地方。

另一個相關但不同的問題：`patches.py` 的 `_compute_ee_relative_stats` 日誌
訊息寫的是 `n_offset_steps=%d`（計畫提議的改名目標），但底層變數仍然叫
`action_delta_indices`——改名到達了日誌文字，卻沒有到達那一處的程式碼
識別字。

---

# 與 `claude_docs/ee-delta-flow-plan.md` 的偏差

1. **mcap_converter 的逐手臂分支結構**（`extractor.py:1357-1424`）與計畫的
   字面描述不同（「只有 `action_slices.append(...)` 這一步計算會分支到
   delta 公式」）：實際上 delta 呼叫只發生一次，在逐手臂迴圈*之後*，對整個
   串接後的陣列進行——功能上是等價的（因為 `ee_delta_forward` 內部本來就會
   逐手臂迴圈），但結構上與計畫的描述不同，若你是照著計畫的行號在讀 diff，
   要留意這一點。
2. **沒有任何已交付的設定曾經設定 `ee_action_encoding: "delta"`。** 計畫的
   第 4 項要求用新旗標轉換一個真實 session；經驗上檢查，repo 中沒有任何
   一份 YAML（不論是已交付的設定或測試固件）設定了這個欄位——delta 路徑
   只透過直接建構 `DataConfig` 的 Python 單元測試被演練過。**第 4 項
   （實際的轉換＋訓練步驟）在本分支上尚未被執行／接上。** 這是整個
   程式庫中最具體的一個「實際上還剩什麼待辦」訊號。
3. **`docs/data-conversion.md` 與 `docs/training.md` 都沒有記載這個新
   機制。** mcap_converter 的文件完全沒有提到 `ee_action_encoding`／delta
   烘焙；`docs/training.md` 的 action-type 表格列出了
   `joint_abs`/`ee_abs`/`ee_relative`，卻遺漏了 `ee_delta`——即使
   `train.py` 自己的 `_ANVIL_HELP` 文字已經完整記載了它。這讀起來像是
   一次不完整的收尾，而不是刻意的省略（沒有任何地方標示這是待辦事項）。
4. **GT-replay（第 3 項）並未使用 `EvaluationDataset`**，儘管計畫明確指示
   「重用 `anvil_shared.ee_transform` ＋
   `anvil_eval.dataset.EvaluationDataset`」——它改為直接手動讀取
   parquet／`info.json`，這是一個刻意的選擇（有記載的理由：獨立於訓練時的
   transform），但確實是一個真實的偏差，讓資料集載入邏輯在程式庫中重複
   存在了兩份。
5. **第 5 項（離線評估，「門檻降低」）對 `ee_delta` 而言完全沒有實作**——
   這是本次發現中最具影響力的一項偏差。`evaluator.py`、`metrics.py`、
   `plotting.py`、`anvil_eval_ros/cli.py` 都用會遺漏 `"ee_delta"` 的元組
   來判斷 EE 模式。計畫的門檻是「能跑起來、有輸出，FAIL 沒關係」——實際
   發生的情況比 FAIL 還糟：一個 `ee_delta` checkpoint 會悄悄走入**通用、
   非 EE 的指標路徑**，產出一個把公尺、無單位 rot6d、夾爪單位混在一起的
   無意義 MAE/MSE，且完全沒有任何警告表明這件事發生了。程式碼中沒有任何
   地方承認這是一個已知缺口。
6. **超出計畫範圍、未被記載為決定的擴大範圍：** 整個舊有關節空間
   `delta_obs_t`/`delta_sequential` 機制被移除（`delta_restore.py` 刪除、
   `eval_recorder_node.py` 的 GT 重建刪除、`action_limiter.py`／
   `docker-compose.eval.yml`／`plot_monitor_csv.py` 的清理）——真實、可觀，
   且計畫中從未提及有做過這個決定。
7. **與計畫完全一致，經直接讀取確認**（依使用者要求，不論哪個方向都要
   明確標示出來，不能悄悄抹平）：`ee_delta_forward`/`ee_delta_inverse`/
   `ee_delta_restore_step` 中的世界座標系旋轉數學公式；第一幀自我錨定慣例
   端到端一致（converter → 訓練統計量 → GT-replay）；epsilon 下限與
   rot6d-identity 夾限這兩個統計量防護機制；明確的 COMPLETED/FAILED
   統計量日誌；第 2b 項假硬體回聲／積分行為；以及 GT-replay 機器精度等級
   的嚴格性、沒有針對煙霧測試規模放寬。

---

# 已知臭蟲、缺口與粗糙之處（彙整，大致依嚴重程度排序）

**很可能會崩潰／正確性臭蟲：**
1. **`self._obs_lock` 對經典（非 VLA）模型從未初始化**（`inference_node.py`）
   ——只在 `_setup_vla_inference()` 內部建立，只在 `self._is_vla` 為真時
   才會呼叫。但 `_obs_update`/`_publish_loop` 在 `self.is_ee_delta` 為真時
   一律無條件執行 `with self._obs_lock:`——而 Diffusion（ee_delta 的目標
   架構）永遠不會是 `_is_vla`。幾乎可以確定會在任何真實 ee_delta
   checkpoint 第一次 `_obs_update` tick 時拋出 `AttributeError`。
   **尚未透過實際執行來驗證——這是在信任這整套機制真的能端到端運作之前，
   最優先該檢查的一項。**
2. **`ee_delta` 在至少五個地方各自獨立地被悄悄誤導或丟棄**：
   `config.py:343`（訓練輸出目錄路由 → 落到
   `model_zoo/joint-space/` 而不是 EE-space）、`config.py:543`
   （`validate_action_space`——完全沒有資料集形狀驗證）、
   `eval_recorder_node.py`（落入關節模式主題訂閱）、`evaluator.py`/
   `metrics.py`/`plotting.py`（落入無意義的通用指標）、
   `anvil_eval_ros/cli.py`（落入關節模式評估設定產生）。這五處都沒有拋出
   錯誤或記錄警告——全都悄悄地做錯事。
3. **沒有任何已交付的設定演練過 delta 烘焙路徑**——第 4 項從未真正透過
   實際 CLI 執行過；只透過直接建構 `DataConfig` 做過單元測試。

**真實但嚴重程度較低的缺口：**
4. `anvil_config.json` 遺漏 `is_ee_delta`（checkpoint metadata 與類似的
   `is_ee_relative` 欄位不一致）。
5. 兩個可能會逐漸失去同步的「合法 action types」登記處：
   `anvil_shared.action_types.VALID_ACTION_TYPES`（沒有 `ee_delta`）vs.
   `anvil_trainer.config._VALID_ACTION_TYPES`（有）——前者文件上宣稱是
   共用的權威來源，但後者實際上並沒有為此目的匯入它。
6. `anvil_shared/__init__.py` 的 `__all__` 列出了 `ee_obs_abs_forward`
   卻沒有匯入它——若透過套件根層級匯入會拋出 `AttributeError`（目前潛伏，
   沒有任何真實呼叫點會踩到）。
7. 推論端有兩個各自獨立、未同步的 EE 模式偵測器
   （`inference_node.py` 依 checkpoint 判斷的 `resolve_action_type` vs.
   `multi_process.py` 依設定判斷的 `ee_command_topic` 存在性檢查）——
   checkpoint／設定配對錯誤時會悄悄失敗，而不是明確報錯。
8. `multi_process.py` 的 EE 就緒檢查只要求*一支*手臂回報過即可，不要求
   所有已設定的手臂——雙臂設定中某支手臂的發布者掛掉時，該手臂的觀測值
   會被悄悄永遠設為零向量。
9. 4 份陳舊／已損壞的舊版 `configs/mcap_converter/*.yaml`（經驗證，在目前
   loader 下會拋出 `ValueError` 或悄悄產出空的主題）——與 EE-delta 本身
   無關，但確實是同一分支更廣泛的統一設定重構工作留下的堆積雜物。
10. `gt_replay.py` 的第一幀容許誤差混用了單位（一個以度為單位的 CLI 旗標，
    被縮放後拿來當作無單位 L2 量的門檻）。
11. `gt_replay.py` 不產生任何結構化（JSON/CSV）輸出，與套件中其他每個工具
    都不同——PASS/FAIL 只有 stdout 加上結束代碼。

**重複／違反 DRY 原則（目前對正確性無害，但未來有漂移風險）：**
12. 「每手臂 10 維，`[x,y,z,r0..r5,grip]`」這個動作版面常數，至少在四個
    地方各自被寫死（`gt_replay.py`、`plotting.py`、
    `anvil_eval_ros/cli.py`、隱含在 `metrics.py` 中），沒有一處匯入
    `anvil_shared` 中的共用常數。
13. 測地線旋轉誤差公式在兩個檔案中以兩種不同方式實作
    （`gt_replay.py` 以四元數為基礎的 `2*arccos(|qw|)` vs. `metrics.py`
    以矩陣跡為基礎的 `arccos((trace-1)/2)`）——數學上等價，但從未共用過。
14. PASS/FAIL 門檻（`0.02` 公尺／`0.0873` 弧度 ≈ 5°）在
    `metrics.py`（兩處）與 `reporting.py`（第三種單位慣例）中被複製貼上
    3-4 次。
15. Checkpoint 路徑解析邏輯在 `inference_node.py` 與 `ee_runtime.py` 中
    重複（後者自己的 docstring 已承認這是刻意但未共用的鏡射）。
16. `ee_action_encoding` 驗證在 `loader.py`（拋出 `ValueError`）與
    `validators.py`（拋出 `ConfigurationError`）中重複實作——後者在目前
    實際呼叫鏈中是死碼。
17. `gt_replay.py`/`anvil_eval/cli.py`/`anvil_eval_ros/cli.py` 中有三份
    幾乎相同的 `setup_logging()` 樣板程式碼。
18. `patches.py` 中多餘的統計量修改：`_compute_ee_delta_stats` 就地修改
    `full_dataset.meta.stats`（第 512、534 行），但真正對訓練有影響的是
    `train_dataset`（另一個經過篩選的實例）——被呼叫端稍後所做的修改
    （patches.py:826-828）才是真正有效的那一次；第一次修改讀起來像是
    有作用，實際上沒有。

**命名／API 形狀上的小瑕疵（外觀層面，但確實會讓第一次閱讀的人感到困惑）：**
19. `_classic_action_deque` 現在依模式不同、裝載著語意完全不同的內容——
    `ee_delta` 是原始 delta，其他模式是已還原的絕對值——但名稱或型別都沒有
    表明這一點。
20. `ee_relative_restore_chunk`/`ee_delta_restore_step` 概念上是對等的
    （「依照某個參照，把某種編碼還原成絕對值」），卻沒有共用的基底類別／
    介面，也沒有任何機制阻止用一個批次去呼叫那個單步函式（若被誤用，會
    悄悄重現 n-0 式的過舊問題）。
21. `EEAbsTransform`/`EEDeltaTransform` 未在 `anvil_trainer` 套件層級
    匯出，不像結構上完全對等的 `EERelativeTransform`。
22. `rotation.py` 的批次版本 `rot6ds_to_matrices` 對退化輸入做夾限，而
    純量版本 `rot6d_to_matrix` 對同樣情況卻拋出例外，docstring 的理由讀起來
    是反過來的（夾限通常正是*掩蓋*臭蟲的做法，而不是避免掩蓋）——早於本分支
    就存在，但直接關係到稽核 `ee_delta_*` 的數值行為。
23. `scripts/training_metrics.sh` 內部使用橫幅仍自稱
    `benchmark_training.sh`——一個改名遺留的不一致之處（不論如何，這個
    腳本都與 EE-delta 無關）。

---

# 附錄 — 本分支改動、但不屬於七個 EE-delta 項目的檔案

在研究過程中已確認並排除在上方深度說明之外，列在此處以免完整分支稽核時
重新花時間調查它們是不是 EE-delta 工作的一部分：

- `ema.py`、`test_umi_features.py`、`scripts/training_metrics.sh` ——
  EMA/DDPM-IP/DDIM-default 訓練品質改進包，是對診斷報告中未解 EMA 問題的
  直接回應，與 delta 表示法工作彼此正交。
- `image_worker.py`、`run_inference.sh` 中的 ffmpeg 批次轉檔新增內容、
  `docker/inference/Dockerfile`/`entrypoint.sh` —— 推論監控用的全 episode
  JPEG／影片錄製功能。
- `mcap_player_node.py` —— 死碼刪除，與 EE 無關。
- `packages/mcap_converter/src/mcap_converter/cli/inspect.py` ——
  獨立的 MCAP 主題／schema 檢視 CLI，沒有引用任何 EE-transform。
- `configs/mcap_converter/openarm_bimanual_quest.yaml`、
  `openarm_bimanual.yaml`、`openarm_single_quest.yaml`、
  `openarm_single_quest_afo.yaml` —— 陳舊的舊格式設定，在目前 loader 下
  經驗證已損壞；屬於本分支更廣泛的統一設定重構工作留下的無關雜物。
- `configs/cyclonedds/*` 改名（`two_pc_gpu.xml`→`gpu_pc.xml` 等）以及
  `docs/inference.md` 中「Distributed Inference Architecture」一節的
  改寫 —— 與 EE-delta 無關的文件／基礎設施整理工作，只是剛好動到了部分
  同樣的檔案。
- 大量相對於 `main` 為 `D`（已刪除）狀態的
  `packages/mcap_converter/.../{dataset_viz.py, mcap_valid.py,
  quality.py, schema_inspect.py, viz/}` 及其對應測試 —— 本分支的工作樹
  單純早於 `main` 上存在的這些工作；不是本分支為了 EE-delta 工作而刪除的
  內容（是分支分歧，不是設計決策）。
