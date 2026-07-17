# ee_rel Diffusion Policy 推論失敗根因分析

> **狀態**：診斷完成，待實作修正  
> **日期**：2026-06-18  
> **作者**：Patrick Hsu（Claude Code 協助分析）  
> **症狀**：`ee_rel` checkpoint 推論出現「隨機、亂飛」動作，完全未嘗試接近物體
>
> **命名說明（後補）**：本文件保留為當時（修正前）的診斷記錄，通篇以 `ee_rel`（此機制在成文當時的名稱）指稱。此機制已於 2026-06-22 修正並在其後改名為 `ee_relative`；`ee_rel` 現為同一 action_type 的永久別名。不反映當前行為，僅作根因診斷歷史記錄。

---

## 摘要

在對照 UMI（Universal Manipulation Interface）、Diffusion Policy 基線與本實作三方程式碼後，**使用者的錨點不匹配假設已獲驗證**：

> **根本問題（H1）**：UMI 的設計讓 observation 與 action 使用「同一個當前 EE pose」做錨點，並採完整 SE(3) 轉換；本實作的 `ee_rel` 模式卻是「絕對世界座標 obs」搭配「相對當前態 action」，obs 與 action 的錨點/座標系完全不一致。加上 obs 以絕對世界座標的 MEAN_STD 正規化，部署時只要機器人位置或起始姿態略微偏離訓練分佈，obs 即成為 out-of-distribution 輸入，Diffusion Policy 在 OOD 條件下去噪，輸出垃圾 action。

---

## 目錄

1. [UMI 的相對 Pose 處理](#1-umi-的相對-pose-處理)
2. [Diffusion Policy Baseline 的 Action 處理](#2-diffusion-policy-baseline-的-action-處理)
3. [本實作的 ee_rel 管線](#3-本實作的-ee_rel-管線)
4. [三方對照表](#4-三方對照表)
5. [根因分析與建議修正](#5-根因分析與建議修正)

---

## 1. UMI 的相對 Pose 處理

> **關鍵結論**：UMI 把 observation（本體感知）與 action **同時**表示為相對於「同一個當前 EE pose 錨點」的軌跡，並採完整 SE(3) 變換（平移與旋轉都轉入 body frame）。

### 1.1 設定

```yaml
# universal_manipulation_interface/diffusion_policy/config/task/umi.yaml:87-89
pose_repr: &pose_repr
  obs_pose_repr: relative   # 'abs' 或 'relative'
  action_pose_repr: relative
```

`relative` 模式的數學（與已標為 buggy 的 `rel` 模式不同）：

```python
# universal_manipulation_interface/diffusion_policy/common/pose_repr_util.py:62-64
elif pose_rep == 'relative':
    out = np.linalg.inv(base_pose_mat) @ pose_mat
    return out
```

即：`T_relative = T_base^{-1} @ T_pose`。由於矩陣乘法在齊次座標下同時旋轉平移向量，平移也被轉入 body frame，不只是世界座標減法。

> ⚠️ 注意：UMI 還有一個 `'rel'` 模式（`pose_repr_util.py:53-61`），是「平移做世界座標相減、旋轉做 `R @ inv(R_base)`」，程式碼中明確標為 `# legacy buggy implementation / for compatibility`。當前所有 config 皆使用 `'relative'`。

### 1.2 訓練時 obs 與 action 使用同一錨點

```python
# universal_manipulation_interface/diffusion_policy/dataset/umi_dataset.py:337-353
# —— 建立錨點 ——
pose_mat = pose_to_mat(np.concatenate([
    obs_dict[f'robot{robot_id}_eef_pos'],
    obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
], axis=-1))
action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])

# —— obs 轉相對 ——
obs_pose_mat = convert_pose_mat_rep(
    pose_mat,
    base_pose_mat=pose_mat[-1],      # ← 錨點 = 當前 EE
    pose_rep=self.obs_pose_repr,
    backward=False)

# —— action 轉相對，用同一個錨點 ——
action_pose_mat = convert_pose_mat_rep(
    action_mat,
    base_pose_mat=pose_mat[-1],      # ← 同一錨點
    pose_rep=self.obs_pose_repr,     # ← 注意：用 obs_pose_repr，非 action_pose_repr
    backward=False)
```

`pose_mat[-1]` 是觀測 horizon 最後一幀（即「現在」）的 EE pose，obs 與 action 兩者共用。

### 1.3 轉換後覆寫回 obs dict

```python
# umi_dataset.py:356-364
obs_pose = mat_to_pose10d(obs_pose_mat)        # 相對 pose → rot6d 表示
action_pose = mat_to_pose10d(action_pose_mat)

obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]                # 覆寫 obs
obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]     # 覆寫 obs
```

因此餵入 policy 的 obs 本體感知永遠是「以當前 EE 為原點的相對軌跡」，最後一幀（當前）= 恆等變換 ≈ 零值，具有**平移不變性**。

### 1.4 推論時逆轉換使用同一錨點

```python
# universal_manipulation_interface/umi/real_world/real_inference_util.py:182-200
# obs 觀測也用同一錨點正規化
pose_mat = pose_to_mat(np.concatenate([
    env_obs[f'robot{robot_idx}_eef_pos'][-1],
    env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
], axis=-1))

action_mat = convert_pose_mat_rep(
    action_pose_mat,
    base_pose_mat=pose_mat,          # ← 當前 EE（同 obs 所用的錨）
    pose_rep=action_pose_repr,
    backward=True)                   # 逆轉換：base @ pose_mat
```

逆轉換數學（`pose_repr_util.py:94-96`）：`out = base_pose_mat @ pose_mat`，即 `T_abs = T_base @ T_rel`，是 forward 的精確反函數。

### 1.5 正規化策略

訓練時 DataLoader 先做上述 SE(3) 轉換，再計算統計量（`umi_dataset.py:193-227`），所以 stats 反映的是「相對分佈」：
- position（已相對化，值接近 0）→ `get_range_normalizer_from_stat`，映射到 `[-1,1]`
- rotation rot6d → `get_identity_normalizer_from_stat`（已有界，留原值）
- gripper width → `get_range_normalizer_from_stat`

**已驗證 ✓**

---

## 2. Diffusion Policy Baseline 的 Action 處理

> **關鍵結論**：Diffusion Policy 的 observation **永遠是絕對值**；action 可為相對（逐步 OSC delta，由 delta controller 直接消費）或絕對（`abs_action=True`），兩種模式皆被驗證可用。**沒有任何模式是「相對 obs + chunk 相對 action 由 policy 負責還原為 waypoint」的組合。**

### 2.1 obs 恆為絕對

```python
# diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py:141-145
def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer):
    obs = np.concatenate([
        raw_obs[key] for key in obs_keys
    ], axis=-1).astype(np.float32)
    # abs_action 參數僅影響 action 分支，obs 不受影響
```

obs key 包含 `robot0_eef_pos`、`robot0_eef_quat`、`robot0_gripper_qpos`——皆是感測器直接讀取的絕對世界座標（`square_lowdim.yaml:11-18`）。

### 2.2 action 表示兩種模式

**相對模式（預設，`abs_action=False`）**：

```python
# diffusion_policy/dataset/robomimic_replay_image_dataset.py:223-242
def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions   # 直接使用原始 OSC delta action，不做任何轉換
    if abs_action:
        ...  # 僅在 abs 模式才做轉換
    return actions
```

`abs_action=False` 時，action 是 robomimic OSC controller 的 7-dim delta（`Δpos(3) + Δaxis-angle(3) + gripper(1)`），直接餵給 `control_delta=True` 的 controller 執行，**不需要 policy 還原為絕對 waypoint**。

```python
# diffusion_policy/env_runner/robomimic_image_runner.py:306-310
env_action = action
if self.abs_action:
    env_action = self.undo_transform_action(action)
obs, reward, done, info = env.step(env_action)
```

**絕對模式（`abs_action=True`）**：obs 和 action 皆為絕對，policy 輸出絕對 waypoint，`undo_transform_action` 把 rot6d 還原為 axis-angle，再以 `control_delta=False` 的 controller 執行（`robomimic_image_runner.py:85-87`）。

### 2.3 obs 正規化

- `eef_pos`：`get_range_normalizer_from_stat`，對**絕對**座標做 min-max → `[-1,1]`
- `eef_quat`：`get_identity_normalizer_from_stat`（已在 `[-1,1]`）
- `abs_action=False` 時 action 正規化：`get_identity_normalizer_from_stat`（OSC delta 假設已在 `[-1,1]`）

**已驗證 ✓**

---

## 3. 本實作的 ee_rel 管線

> **關鍵結論**：obs 為絕對世界座標，action 為相對當前態的「混合 frame 表示」（平移 world frame + 旋轉 body frame），obs 與 action 錨點不一致，obs 以絕對座標 MEAN_STD 正規化。

### 3.1 轉檔：obs 與 action 來源

```python
# packages/mcap_converter/src/mcap_converter/core/extractor.py:1368-1375
_, pos, quat, gripper = buffer[idx]
rot6d = matrix_to_rot6d(quat_to_matrix(quat))
state_slices.append(
    np.concatenate([pos, quat, np.array([gripper], dtype=np.float64)])
)   # observation.state: [x,y,z, qx,qy,qz,qw, gripper]  ← 絕對世界座標

action_slices.append(
    np.concatenate([pos, rot6d, np.array([gripper], dtype=np.float64)])
)   # action: [x,y,z, rot6d(6), gripper]               ← 也是絕對，轉換在下游做
```

obs.state 與 action 皆來自同一個 `/ee_pose_<arm>` 訊息（CommandedEEPose），格式為 `frame_id="world"` 的絕對姿態（`CommandedEEPose.msg:1-7`）。action 在轉檔階段是絕對值，「轉相對」由訓練 transform 在後做。

### 3.2 訓練 transform：只改 action，不動 obs

```python
# packages/anvil_trainer/src/anvil_trainer/transforms.py:195-228
def apply(self, item: dict) -> dict:
    state = item["observation.state"]       # L203 — 讀 obs
    ...
    anchor = state[-1]                       # L206-207 — 用最後一幀做錨點
    ...
    delta_np = ee_rel_forward(              # L223 — 只轉換 action
        action_abs_np, state_np if self.use_per_sample_state else anchor_np
    )
    item["action"] = torch.tensor(delta_np) # L228 — 只寫回 action
    # observation.state 從未被改寫                ← 關鍵問題
    return item
```

`EERelTransform.apply` 只修改 `item["action"]`，`item["observation.state"]` 保持原始絕對世界座標，原樣進入 MEAN_STD 正規化。

### 3.3 ee_rel 數學：混合 frame 表示

```python
# packages/anvil_shared/src/anvil_shared/ee_transform.py:108-135（ee_rel_forward）
for arm in range(n_arms):
    # 平移：world/base frame 相減
    result[..., a0:a0+3] = action_abs[..., a0:a0+3] - state_xyz     # ← world frame

    # 旋轉：body frame（R_state.T @ R_action）
    Rs_rel = Rs_state_T @ Rs_action                                   # ← body frame
    result[..., a0+3:a0+9] = matrices_to_rot6d(Rs_rel)
    # gripper 保持絕對
```

- 平移 delta：`act_pos - state_pos`，仍在**世界座標系**（world frame）。無旋轉不變性。
- 旋轉 delta：`R_state.T @ R_act`，在**body frame**。

對照 UMI `'relative'`（`inv(base) @ pose`）：這個全矩陣乘法會同時旋轉平移向量，使平移也進入 body frame。本實作平移用世界座標相減、旋轉用 body frame，是 UMI `'rel'`（legacy buggy）的平移 + UMI `'relative'` 的旋轉的混合，不對應任何已驗證的 UMI 設定。

round-trip 自洽性：17 個測試通過，`ee_rel_inverse(ee_rel_forward(a,s), s) == a` 在數學上成立——**混合表示在編碼/解碼上是自洽的，數學本身不是 bug**。

### 3.4 正規化：obs 用絕對座標的 MEAN_STD

Checkpoint `config.json` 的 `normalization_mapping`：
```json
"ACTION": "MIN_MAX",
"STATE":  "MEAN_STD",
"VISUAL": "IDENTITY"
```

`patches.py` 的 `_compute_ee_rel_stats`（L144-242）只 patch `stats["action"]`（L231、L412），`stats["observation.state"]` 完全未修改，保留 dataset 計算的絕對座標 mean/std。

> 含義：推論時若機器人位置或基座原點偏離訓練工作區哪怕幾公分，obs.state 正規化後的值就會 OOD。

### 3.5 推論端 obs 餵入

```python
# ros2/src/lerobot_control/lerobot_control/strategies/multi_process.py:183-189
self._ee_state_by_arm[name] = [
    p.x, p.y, p.z,
    o.x, o.y, o.z, o.w,
    msg.gripper,
]
```

```python
# multi_process.py:259-267
state_flat: list[float] = []
for arm_name in self._ee_arm_order:
    state_flat.extend(self._ee_state_by_arm.get(arm_name, [0.0] * 8))
observation["observation.state"] = torch.tensor(state_flat, ...)
```

絕對世界座標，無任何相對轉換，原樣送入 policy。

**已驗證 ✓**

---

## 4. 三方對照表

| 面向 | UMI（SOTA，可用）| Diffusion Policy Baseline | 本實作 ee_rel（失敗） |
|---|---|---|---|
| **obs 表示** | **相對**，錨於當前 EE `pose_mat[-1]` | 絕對世界座標 | **絕對**世界座標 |
| **action 表示** | 相對，**同一** `pose_mat[-1]` 錨 | 相對逐步 delta（delta controller 直接消費）/ 或絕對 | 相對當前態（chunk 整段錨於單一當前態） |
| **obs ↔ action 錨點** | **完全一致** | 不適用（action 由 controller 消費，非 policy 還原） | **不一致**（obs 絕對 / action 相對） |
| **平移 frame** | body（`inv(base)@pose` 把平移轉入 body） | controller 處理 | **world/base**（`act_pos - state_pos`） |
| **旋轉 frame** | body | controller 處理 | body（`R_state.T @ R_action`） |
| **obs 正規化** | range[-1,1]，分佈為相對值（值近 0）| range / abs-max，對絕對座標 | **MEAN_STD 對絕對座標** |
| **平移不變性** | ✅ 有（obs 永遠錨於當前態）| ➖ controller 處理 | ❌ 無（obs 為絕對） |
| **旋轉不變性** | ✅ 有（同 SE(3) 錨）| ➖ controller 處理 | ❌ 無（obs 絕對姿態） |
| **action 還原方式** | policy chunk 輸出 → `base @ rel` → 絕對 waypoint | delta controller 直接執行 | policy chunk 輸出 → `ee_rel_inverse` → 絕對 waypoint |
| **已驗證可用** | ✅ UMI 論文 + 實際部署 | ✅ robomimic 基線 | ❌ 實機亂飛 |

---

## 5. 根因分析與建議修正

### 5.1 根因清單（依可能性排序）

---

**H1（主因）— 絕對 obs + chunk 相對 action 的錨點不匹配**（已驗證）

UMI 之所以穩健，在於：
1. obs 永遠錨於當前態，值恆近 0，具平移不變性
2. policy 學到「相對 obs 軌跡 → 相對 action 軌跡」，兩者在同一 body frame 座標
3. 正規化統計量反映相對分佈（小值 + 緊湊），對工作區位置不敏感

本實作：
1. obs 為絕對世界座標，正規化用絕對座標的 MEAN_STD → 部署時只要機器人/基座位置略偏，obs 即 OOD
2. action 為相對當前態，policy 學的是「絕對 obs → 相對 action」這個 UMI 和 DP 皆未驗證過的組合
3. Diffusion Policy 在 OOD obs 條件下去噪 → 輸出垃圾 action → 「隨機亂飛、不嘗試接近物體」

**直接證據**：
- `transforms.py:228`（只改 action，不動 obs）
- `patches.py:231,412`（只 patch action stats）
- `multi_process.py:183-267`（推論餵入絕對 obs，無相對轉換）
- `umi_dataset.py:343-353`（UMI obs + action 同錨）

---

**H2（加重因子）— 混合 frame 表示（world 平移 + body 旋轉）**（已驗證為設計問題，但非亂飛唯一原因）

`ee_transform.py:116`（平移 world frame）vs `ee_transform.py:126`（旋轉 body frame）是已知設計，round-trip 自洽。此混合表示：
- 平移缺旋轉不變性（機器人轉向後，同一目標點的平移 delta 改變）
- 表示缺乏幾何一致性，學習難度更高
- 不對應任何已驗證的 UMI `rel` / `relative` / `delta` 模式

**結論**：此為加重因子，使 policy 更難學習，但不是「完全不嘗試接近」的單一致命原因。

---

**H3（次要設定問題）— `n_action_steps: 12` vs 訓練值 8**（假設，待驗證）

Checkpoint `diffusion_20260612_101136` 訓練用 `n_action_steps=8`（`action_delta_indices = range(-1, 23)`，即 horizon=24）。`inference_ee.yaml:52` 設定 12。

- 12 仍在 horizon 24 內，不越界，故不是「亂飛」主因
- 但 open-loop 執行 12 步（訓練只用 8 步）可能加速漂移累積
- **假設**：對一個已能正常推論的模型，此設定值得修回 8；但對目前已 OOD 的 obs 輸入，此設定不是主要問題

---

**DP 反證（誠實記錄）**

Diffusion Policy 預設組態（`square_image.yaml`，`abs_action=False`）亦是「絕對 obs + 相對 action」卻可用——這似乎與 H1 矛盾。

**關鍵差異**：DP 的相對 action 是**逐步 OSC delta，由 delta controller 即時消費**，policy 只決定「下一步往哪移動 Δ」，不需要還原為未來 n 步的絕對 waypoint。本實作採用 UMI 式「整段 chunk 錨於當前態、還原為未來 8 步的絕對 waypoint、開環執行」，兩者的誤差累積特性完全不同。

> 因此 H1 的真正問題是：**把 UMI 式 chunk 相對 action（需 obs 也相對才能正常學習）接上了 DP 式絕對 obs（對 UMI 式 chunk 推論是 OOD）**，這個組合兩個已驗證參考皆未採用。

---

### 5.2 建議修正（下一輪實作，需重新轉檔/重訓驗證）

#### 方案 A — 對齊 UMI（根治，推薦）

**目標**：obs 也改為相對——錨於當前 EE，採完整 SE(3) 轉換（`inv(base) @ pose`），使 obs 與 action 同錨同 frame，消除混合表示的幾何不一致。

需修改的位置：

1. **轉檔端** — `packages/mcap_converter/src/mcap_converter/core/extractor.py`  
   `_align_ee_signals`（約 L1368）：在組裝 `state_slices` 時，改為輸出相對 pose（錨於當前態 `inv(base) @ pose`，轉成 rot6d），或增加 EE 相對 obs 模式。

2. **訓練 transform** — `packages/anvil_trainer/src/anvil_trainer/transforms.py`  
   `EERelTransform.apply`：在寫回 `item["action"]` 的同時，也對 `item["observation.state"]` 做 SE(3) 相對轉換（用同一個 `anchor`）。此時 obs.state 的旋轉表示需從 quaternion 改為 rot6d（或改為 axis-angle，配合 UMI 設計）。

3. **ee_transform.py** — `packages/anvil_shared/src/anvil_shared/ee_transform.py`  
   `ee_rel_forward` / `ee_rel_inverse`：把平移從 world frame 改為 body frame（對齊 `'relative'` 模式），消除混合表示。

4. **推論端 obs** — `ros2/src/lerobot_control/lerobot_control/strategies/multi_process.py`  
   `_build_observation`：在組裝 `observation.state` 時，對當前 EE pose 做 SE(3) 相對轉換（錨於最新觀測的 `[-1]` 幀）。

5. 重新轉檔 + 重訓，驗證實機行為。

#### 方案 B — 主力 ee_abs，暫停 ee_rel（務實，短期）

`ee_abs` checkpoint（`diffusion_20260611_141344`）已可正常到達物體並抓取。暫時以 `ee_abs` 為主力，`ee_rel` 列為待修復，等方案 A 實作完成並驗證後再啟用。

---

### 5.3 修正優先序

| 優先 | 項目 | 影響 | 工作量 |
|---|---|---|---|
| P0 | 確認夾爪力道（Issue 1，`gripper_factor` 已實作）| 立即 | ✅ 已完成 |
| P1 | 方案 A：obs 改相對 + 消除混合 frame，重轉檔重訓 | 根治 ee_rel | 高（需重新 pipeline） |
| P2 | 修正 `n_action_steps` 回訓練值 8 | 次要 | 低（config 一行） |
| P3 | 方案 B：短期主力 ee_abs，收集更多 ee_abs demo | 短期可用 | 低 |

---

## 附錄：關鍵程式碼引用索引

| 檔案 | 關鍵行 | 說明 |
|---|---|---|
| `umi_dataset.py` | L343-353 | obs + action 同錨點 `pose_mat[-1]` |
| `pose_repr_util.py` | L62-64 | `'relative'` = `inv(base)@pose` |
| `real_inference_util.py` | L101-117, L182-201 | 推論端 obs/action 同錨逆轉換 |
| `umi.yaml` | L87-89 | `obs_pose_repr: relative` |
| `transforms.py` | L202-228 | `EERelTransform` 只改 action，不動 obs |
| `patches.py` | L231, L412 | 只 patch action stats |
| `multi_process.py` | L183-189, L259-267 | 推論餵入絕對 obs |
| `ee_transform.py` | L108-135, L138-202 | 混合 frame：平移 world + 旋轉 body |
| `extractor.py` | L1368-1375 | 轉檔：obs 與 action 皆為絕對，同一來源 |
| `robomimic_replay_image_dataset.py` | L223-243 | DP `abs_action=False` = raw delta |
| `robomimic_image_runner.py` | L85-87, L306-310 | DP delta controller |
