# 計畫 — 在真實 OpenArm 硬體上實作 Delta-flow (n-(n-1)) EE 笛卡爾管線

> 本文為 `claude_docs/ee-delta-flow-plan.md` 的繁體中文翻譯版本。程式碼識別字、函式名稱、
> 設定欄位、`action_type` 字串、檔案路徑與行號等技術性內容維持原文，不做翻譯。

## 背景

先前的診斷報告（`claude_docs/ee-space-libero-vs-production-diagnosis.md`，位於本
worktree）指出，正式環境中 `ee_rel` 在真實硬體上出現的抖動（jitter），成因是 **n-0
逐 chunk 相對化**（整個預測 horizon 都相對於同一個固定的 chunk 起始錨點做相對化；
normalization 統計量是跨整個 horizon offset 一起彙整的 → 近期／實際被執行的動作在
normalize 後的空間中被壓縮）。LIBERO 自身直接可對照的 `goal-world-n0` 條件也出現了
同樣的病理現象（Diffusion 從 98% 掉到 16%）。唯一經過驗證有效的機制是 **逐幀單步
delta（n-(n-1)）**——每一步的目標都相對於「緊接在前」的真實狀態——這也是三個已驗證
成功的 LIBERO 條件（native / native_rot6d / native_hand_frame）共通的做法。本計畫
將在真實 OpenArm 硬體上建置一條完整的端到端 delta-flow 管線，徹底取代 n-0 機制。

**反面模式（禁止重用／改編／從中取得靈感）：** `native_ctrlgoal[_relconv]` 的尺度轉換
（只解決 robosuite 專屬問題）、`afo_abs`/`afo_relative`（observation-as-action——依
下方術語稽核，`afo_relative` 實際上是**絕對（Absolute）**、命名有誤；決定其「相對」
語意的是傳遞方式，而非目標表示法本身）、`native_abs`（沒有一致的物理單位）、
`native_n0`（依稽核結果，實際上是退化的 **Delta (n-(n-1))**，命名有誤；它的
「n0」並不表示測試過 chunk-錨點式的 Relative）,以及整個 goal 家族／`goal-world-n0`
（真正的 **Relative (n-0)**——已診斷出的根因機制——應予取代，而非修補）。**不存在
OAA／action_from_observation 路徑**——資料中確實有真正記錄下來的指令（見下方
「Data findings」）；若任何步驟看似需要 OAA，應立即停止並提出警示。

## 兩個 worktree 探索所得的已驗證事實

**資料格式（參考 session `data/raw_sessions/pbib-standard-env-1st-try`，共 251
episodes）——修正了兩個提示假設：**
- `/ee_pose_left`、`/ee_pose_right` 主題的型別是 `anvil_msgs/CommandedEEPose`
  （約 90 Hz）：`header + geometry_msgs/Pose + float64 gripper`，
  `frame_id="world"`。關節狀態為 500 Hz；4 支相機約 60 Hz。
- **是世界座標系（World-frame），不是本體座標系（Body-frame）**（extractor.py:1312-1328
  並未套用任何座標系轉換）。世界座標系與 `native`（已驗證最強的條件）一致，因此沒有問題。
- **`observation.state` 與 `action` 來自同一個 `/ee_pose` 樣本**，差別只在旋轉編碼
  （state = 四元數 8 維／手臂；action = rot6d 10 維／手臂）；位置與夾爪值完全相同。
  轉換器將此 EE 模式標記為「本質上等同於 act-from-obs」（extractor.py:586,
  605-614）。這不是兩個獨立訊號，也**不是**退化的 OAA 後備方案。
- **這個值是「量測」出來的，不是「指令」值**（儘管型別名稱叫 `CommandedEEPose`）：
  `/ee_pose_<arm>` 唯一的發布者（anvil-workcell 的 quest_teleop_controller，
  `absolute_control_modality.py:187-195`）發出的是 TF
  `lookup_transform("world", tcp_link)` = 由量測關節角度做正向運動學算出的結果。
  訓練與推論兩端使用的都是這個量測姿態（一致）。→ delta-mode 發布迴圈的「量測姿態
  自我修正」（條件 a）因此是成立的。逐幀 delta 目標 = 實際達成的量測→量測動作
  （對於一個絕對目標位置控制器而言，這是自洽的）。
- **相依性警示：** `/ee_pose_<arm>` 只有一個發布者（quest_teleop_controller，位於
  另一個 anvil-workcell repo）。第 2 項「每次發布時都用最新的觀測值」這一設計，
  假設它在自主推論期間也會同時被啟動並持續發布——請將此列為明確的飛行前檢查項目
  （若該主題資料過舊，會在不知不覺中破壞自我修正機制）。
- **目前這個 session 尚不存在任何已轉換的 EE 資料集**（`data/datasets/ee-space/`
  為空）。

**可重用的程式碼：** `anvil_shared.ee_transform.ee_rel_forward`（ee_transform.py:68）／
`ee_rel_inverse`（:141）／`ee_obs_rel_forward`（:210），已在合成資料上做過單元測試
（tests/unit/anvil_shared/test_ee_transform.py）。`ee_rel_forward` 已有逐樣本錨點
分支（state.ndim>1）。訓練入口 `anvil-trainer`（train.py:260）只是 lerobot 的一層
薄包裝；新的 action_type 需要動到 5 個地方（config.py 的 `_VALID_ACTION_TYPES`:71
＋ `is_*`:142-152；transforms.py 中新的 Transform 子類別——**依下方重新設計後的
第 1 項，這現在改為模仿 `EEAbsTransform` 只處理 obs 的路徑**（transforms.py:238-259），
**而非 `EERelTransform`，因為 action 已在轉換時就烘焙好，observation 仍維持絕對值**；
patches.py:145-150 的 TransformRunner 清單修補 ＋ :619-628 的 stats 分派；
evaluator.py:194-206 的 eval 反轉分支 ＋ ROS 端的直通處理）。離線評估工具 `anvil-eval`
（cli.py:91）已經可以將某個 checkpoint 對資料集重播，套用 ee_rel 反轉，並計算
位置(m)／方向(度)／夾爪 PASS/FAIL 指標（metrics.py:65）。`anvil-eval-ros`
（cli.py:572）則是在 ROS-in-the-loop 情境下對 MCAP 重播，以貼近實際部署。
**目前沒有任何不依賴模型的 GT-replay 工具**——必須新建（重用 ee_transform ＋
`anvil_eval.dataset`）。轉換器：`mcap-convert` ＋
`configs/mcap_converter/openarm_ee_bimanual_16x9.yaml`。

**推論架構（inference_node.py）：** 目前 `_obs_update`（依 control_freq 頻率執行）
會依序做 get_observation → 觀測相對化 → select_action → **相對於固定的
`_delta_ref_state` 做 ee_rel 還原** → 將絕對動作推入 `_classic_action_deque`
（:806）；`_publish_loop`（:815）只是把絕對值彈出並發布（不讀取觀測、也不做任何
錨定）。目前的還原是相對於一個過時的固定 chunk 錨點（ee_runtime.py:87-128），
也就是 n-0。

**開迴路問題——已由設計決策解決（不是取捨）：** LIBERO 已驗證的條件都是採用逐步
閉迴路執行，delta 是由 robosuite 的 OSC 在每一個執行步驟時，相對於「即時」當前
狀態組合而成的；它們從未做過向前積分，也不曾固定 chunk 錨點
（lerobot_eval.py:165-198；libero_processor.py:460-462）。正式環境將透過下方的
**解耦的 delta-mode 發布迴圈**與此對齊——因此可以在 checkpoint 訓練時所用的完整
`n_action_steps` 下，以開迴路方式執行整個 chunk，完全比照 native 的做法——不需要
`n_action_steps=1` 的節流，也不需要向前積分。

## 設計決策（皆已定案）

- **推論執行方式：** 解耦的 delta-mode 發布迴圈（已定案——見第 2 項）。
- **目標架構：優先選 Diffusion。** 與最初失敗的正式環境架構直接對應，因此本次修正
  可與已診斷出的失敗案例做最直接的比較；LIBERO 的 `native` ＋ 逐幀 delta 在
  Diffusion 上拿到 98-100% 的成功率，因此逐幀 delta 對此架構已獨立獲得驗證。
  （ACT 是自然的後續選項，不在本次範圍內。）
- **Delta 座標系：世界座標系（native 風格）——現已針對平移與旋轉兩者都給出精確定義，
  並對照實際的 robosuite 1.4.0 原始碼驗證過（而非只是從一般慣例近似而來）。**
  詳見下方「旋轉數學（已定案）」一節中的確切公式，以及為何本計畫早先「重用
  `ee_rel_forward`」這句話是錯的。
- **observation.state 的來源（這是事實，不是判斷取捨）：** 已確認訓練與推論兩端
  都是「量測值」——這是直接重新在 `anvil-workcell` 原始碼中追蹤得到的結論（不只是
  從型別名稱去推測）。`/ee_pose_<arm>` 是由
  `absolute_control_modality.py:187-208` 無條件發布的，來源是
  `tf_buffer.lookup_transform("world", tcp_link, ...)`，也就是由真實關節編碼器
  （`joint_state_broadcaster` → `/joint_states` → `robot_state_publisher` TF）
  做正向運動學算出的結果。這段邏輯是在同一個函式中，**先於且獨立於**任何指令目標
  分支執行的，以約 1000 Hz 的計時器，在 teleop 與自主／推論兩種模式下皆是**同一個
  節點、同一段程式碼路徑**（模式差異只影響下游 IK 分支所使用的輸入，不影響
  ee_pose 的發布本身）。發布出來的時間戳是 TF 轉換本身的時間戳，而非 `now()`，
  因此下游的資料過舊問題是可以直接量測的。等價條件 (a) 因此高度可信地成立。

## 六項計畫（每一階段皆以前一階段為前提）

### 旋轉數學（已定案——撰寫第 1 項程式碼前必讀）

已就實際程式碼與 robosuite 1.4.0 原始碼本身做過精確調查（而非近似）。回答了全部
五個子問題：

**1. 現有 `ee_rel_forward`/`ee_rel_inverse` 的公式（原文照引，ee_transform.py）：**
正向旋轉：`Rs_rel = Rs_state_T @ Rs_action`（逐樣本）／`R_state.T @ Rs_action`
（單一 state）——也就是 `R_delta = R_stateᵀ @ R_action`（第 ~124-131 行）。
正向平移：`world_delta @ R_state`，其中 `world_delta = action_xyz - state_xyz`
（第 133 行）——代數上等於 `R_stateᵀ · (action_xyz − state_xyz)`。反向旋轉：
`Rs_abs = Rs_state @ Rs_rel`，即 `R_action = R_state @ R_delta`
（第 ~194-200 行）；反向平移的處理方式類似。**兩個分量依建構方式皆為本體座標系
（BODY-FRAME）**——已在代數上確認兩者互為精確反函數
（`R_state @ (R_stateᵀ @ R_action) = R_action`，因為 `R_state` 是正交矩陣）。

**2. 世界座標系 vs. 本體座標系——修正本計畫先前的框架描述。** 旋轉公式的座標系
慣例，在結構上與平移是相互獨立的（兩行程式碼互不影響）——因此理論上兩者*可以*
混用不同座標系。但現有函式對**兩個分量**都是以本體座標系實作，並非如本計畫先前
暗示的「只有平移是本體座標系」。因此「世界座標系（native 風格）」這項決策需要
**為平移與旋轉都撰寫新公式**，而不是重用旋轉的既有逐樣本分支、只把平移換掉。

**3. 錨點無關性，已確認。** 不論是現有的本體座標系公式，或下方的新世界座標系公式，
都沒有「chunk 起點」或「前一幀」的概念——`state` 只是傳進來的任意陣列。n-0 與
n-(n-1) 之間的差異完全取決於「傳進去當 `state` 的是什麼」（chunk 起始錨點，還是
緊接在前的那一幀），而非公式本身的差異。傳入 `state[t-1]` 就能得到真正的
Delta(n-(n-1))，不會因為公式起源而不慎沿用 n-0 的語意。

**4. 需要新公式的 `ee_delta_forward`/`ee_delta_inverse`——已對照實際 robosuite
1.4.0 原始碼驗證**（`osc.py:261-266`、`control_utils.py:132-136,174-177`——
`native` 這個受驗控制器實際送出指令的地方）：robosuite 本身在
`control_delta=True` 時的組合方式是
`goal_orientation = delta_rotation @ current_orientation`（左乘、世界／外部座標系）
以及 `goal_position = current_position + delta`（直接在世界座標系下相加，不會乘上
目前姿態的旋轉矩陣）。為了精確對齊此行為：
- **正向（在轉換時烘焙）：** `delta_xyz = action_xyz - state_xyz`（單純的世界座標
  差值，不乘上 `R_state`——修正自現有本體座標系公式 `world_delta @ R_state`）；
  `R_delta = R_action @ R_state.T`（世界／外部座標系——修正自現有本體座標系公式
  `R_state.T @ R_action`），透過 `matrices_to_rot6d` 編碼。
- **反向（第 2 項發布迴圈的組合方式，直接回答「精確公式為何」這個問題）：**
  `absolute_target_xyz = obs_xyz + delta_xyz`；`R_absolute_target = R_delta @
  R_obs`——這現在已在代數上確認是精確反函數
  （`(R_action @ R_stateᵀ) @ R_state = R_action`），並且與 robosuite 本身
  完全相同的組合順序一致，因此第 2 項的 `obs_pose ∘ delta_k` 精確對應
  `R_delta @ R_obs`／`obs_xyz + delta_xyz`——沒有不對稱的風險。
- **實務上的意涵：** `ee_delta_forward`/`ee_delta_inverse` 必須撰寫為真正全新的
  函式，而不是既有 `ee_rel_forward`/`ee_rel_inverse` 逐樣本分支的薄包裝。
  可重用的部分是底層的旋轉 PRIMITIVES（`quat_to_matrix`/`quats_to_matrices`、
  `matrices_to_rot6d`、`rot6ds_to_matrices`）——最上層的組合邏輯不同，必須
  重新撰寫。這修正了本計畫先前所有提到「重用 `ee_rel_forward` 逐樣本分支」／
  「包裝既有逐樣本錨點分支」的說法——這些說法現在都應視為被本節取代、不精確。

**5. 第一幀自我錨定（identity/self-anchor）案例，旋轉部分——確認仍然成立，不需要
新的慣例。** 現有測試 `test_identity_state_identity_action_zero_delta`
（test_ee_transform.py:165-176）斷言：當 `state==action`（兩者皆為 identity）時，
`rel[3:9]`（rot6d 維度）在 `atol=1e-12` 下等於 identity 的 rot6d
`[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]`。這個結論可以原封不動地套用到新的世界座標系公式上：
當 `state==action` 時，本體座標系公式 `R_stateᵀ@R_action` 與世界座標系公式
`R_action@R_stateᵀ` 會以完全相同的方式退化為
`R_state@R_stateᵀ = I`——因此自我錨定零 delta 的慣例（第 1 項的第一幀決策）
與座標系慣例無關，不受影響。為新的 `ee_delta_forward` 撰寫一個類似的測試、
斷言相同即可，不需要新的測試架構。

### 第 1 項 — 重新設計：改為轉換時烘焙，而非訓練時的 Transform

**機制變更（已調查並定案）：** 不再於每次 `__getitem__` 時，即時在訓練階段計算
Delta(n-(n-1))，而是由 `mcap_converter` 在轉換時直接把 delta 烘焙進硬碟上的
`action` 欄位——這是一個靜態、可獨立檢視的值，訓練時永遠不會重新計算。這是「花
運算資源之前先驗證數學是否正確」這個原則的更強版本：GT-replay 現在只需要檢查
「硬碟上這個數字對不對」，而不必檢查「每次重新呼叫這段程式碼是否都得到一樣的結果」。

**LIBERO `native` 的先例——已確認，但對原始描述有重要修正。** `native` 的
action **根本不是從絕對姿態計算來的**——它是原始 LIBERO 示範資料中，早已記錄好的
delta *指令* 的原封不動複製（`item["action"].numpy()`，
`libero_convert.py:736-741`；模組 docstring 寫著：「單純的 LOCAL 複製……完全沒有
套用任何 Anvil 轉換」）。`observation.state` 維持原始絕對姿態，未經修改（確認為
「觀測絕對、動作 delta」的不對稱設計）。訓練時完全沒有任何 anvil 轉換介入——就是單純
呼叫 `lerobot-train`，anvil-trainer 完全不參與。**因此 `native` 完全不需要處理
第一幀邊界問題**——它從未對兩個 state 做過相減，所以一開始就不存在 t=-1 的問題
（已確認：整個檔案中唯一與逐幀邊界相關的邏輯，是 `afo_abs` 那個不相關的
*結尾*幀捨棄，`libero_convert.py:803-812`）。

**這為何對正式環境很重要：** OpenArm 並沒有記錄下來的 delta-*指令*通道——
`observation.state` 與 `action` 目前都是從同一個絕對的 `/ee_pose` 樣本推導而來
（本 session 稍早已確認）。因此正式環境烘焙出來的 delta，必然是一種
**差分姿態（differenced-pose）**式的 delta（`action[t] = ee_delta_forward(pose[t],
pose[t-1])`），在精神上與 `native` 相似（單步、逐幀錨點、不做 chunk 累積），但在
建構方式上與 `native` 所驗證的（一個*複製指令*式的 delta）根本不同。**這裡沒有
直接可對照的 LIBERO 先例**——`native` 根本沒遇過第一幀問題。正式環境必須獨立
決定這件事（見下方），而不是「沿用」一個根本不存在的慣例。

**正式環境的烘焙設計（已調查，屬於小範圍局部改動，不是重構）：**
- **mcap_converter**（`packages/mcap_converter/src/mcap_converter/`）：
  `_align_ee_signals`（extractor.py:1341-1380）目前是一個無狀態、逐幀編碼的函式，
  無法取得 `pose[t-1]`，但抽取流程本身早已是嚴格依序逐 episode 進行的
  （`extract_frames`，extractor.py:647-847）——要取得前一幀，只需要額外多接一條
  「管線」（一個 `prev_state` 區域變數，貫穿 `extract_frames`，做法比照既有的
  `next_yield_ts is None` 首值哨兵模式，extractor.py:788-790），不需要改寫迴圈結構。
- **設定：** 在 EE 資料設定上新增一個欄位，例如 `ee_action_encoding:
  "absolute"|"delta"`（預設 `"absolute"`）——**不是**新增一個 `data_space` 值。
  `is_ee` 是依 `data_space=="ee"` 判斷的（schema.py:140-141），每一個 EE 分支
  也都是依 `is_ee` 判斷；若新增一個 `data_space` 值，會動到所有這些地方，並有
  改變既有行為的風險。用一個純量旗標，就能讓 `data_space=="ee"` 維持不變，因此
  既有設定檔（預設為 `"absolute"`）逐位元組完全不變。附帶一提：延伸
  convert.py:713 的輸出目錄後綴（旗標開啟時 `ee-space/`→`ee-delta-space/`），
  讓兩種資料集在硬碟上乾淨地分開。
- **`observation.state`：不變，兩種模式都維持絕對值**（與 `native` 自身的不對稱
  慣例一致，且不需要任何 schema／writer 改動，writer.py 的 feature 宣告本來就是
  依 shape/dtype 通用的，extractor.py:1370-1372 不需要改）。只有
  `action_slices.append(...)` 這一步計算（extractor.py:1373-1375）會分支到
  delta 公式。
- **計算方式：** 呼叫全新的 `ee_delta_forward`（世界座標系公式，依上方「旋轉數學」
  一節——**不是**包裝既有 `ee_rel_forward` 的本體座標系逐樣本分支，依上方修正）。
  底層旋轉 primitives（`quat_to_matrix`/`matrices_to_rot6d`/`rot6ds_to_matrices`）
  仍會重用；extractor 已有對 `anvil_shared` 的延遲匯入（extractor.py:1357）。
- **第一幀慣例（已定案，沒有 LIBERO 先例可循——見上文）：** 採自我錨定零 delta，
  即 `action[0] = ee_delta_forward(action[0], state[0])`，會得到「零平移 delta ＋
  identity 旋轉」的結果。這並非為本計畫臨時發明——它正是既有單元測試
  `test_ee_transform.py:165`（`test_identity_state_identity_action_zero_delta`）
  已經斷言過的不變量，因此這是一個有原則、已經過測試的選擇，沿用 extractor 中
  既有的 `None` 哨兵結構模式（extractor.py:788-790），而不是另立新機制。
- **Writer：不需要改動**——`_define_features`（writer.py:234-238）本來就是依
  shape/dtype 通用宣告 `action`；烘焙出來的 delta 形狀相同，只是數值意義不同。

**烘焙對訓練端的影響（已調查）：**
- **`_first_apply`——已釐清，不論哪種情況都不是真正的風險。** 已追蹤
  `EEAbsTransform`／`EERelTransform` 中每一個讀寫這個旗標的地方：它**只是一次性的
  記錄旗標**（只用來守衛單一一次 `log.info(...)` 呼叫，`transforms.py:335-340`
  等）——實際的數值運算在旗標被檢查之前就已經無條件執行了，而且 transform 是
  每次執行／每個 worker 都重新建立實例的（`patches.py:140`），因此不存在
  跨 worker 的競態，也不存在「第一次呼叫與後續呼叫數值不同」的問題。這個特定疑慮
  不論烘焙與否都已完全排除——採用烘焙仍是基於上述理由（靜態、可檢視的產物，更強的
  「先驗證再運算」原則），而不是因為 `_first_apply` 真的有問題。
- **動作端的 transform 工作被完全省去。** 目前 `apply()` 中即時呼叫
  `ee_rel_forward` 的部分（transforms.py:324-330）以及 `_compute_ee_rel_stats`
  中即時重播＋依 episode 邊界遮罩的統計量重建（patches.py:337-360），都只是因為
  相對化目前是即時進行的——烘焙之後，動作端這兩件事都不再需要。
- **觀測端並非完全省去，但比預期簡單。** 因為 `native` 本身的慣例是讓
  `observation.state` 維持絕對值（不做相對化），新的烘焙 delta 類型在觀測端的
  處理，應該模仿**既有** `EEAbsTransform` 的觀測路徑（`ee_obs_abs_forward`，
  transforms.py:238-259——只做版面轉換，四元數 8n→ rot6d 10n，不做相對化），
  而不是 `EERelTransform` 的觀測路徑（那條路徑還會做相對化）。**這表示新的
  transform 在結構上比較接近 `EEAbsTransform`（只轉換觀測、動作直通），而不是
  一個從零開始的新模式。**
- **統計量大幅縮減，但不會完全消失。** 一個精簡版的 `_compute_ee_delta_stats`
  可以直接從靜態烘焙欄位讀出 mean/std/min/max（不需要即時重播，也不需要依
  episode 邊界遮罩）——但 rot6d ±1 的 identity 夾限（`_force_rot6d_identity`，
  patches.py:60-77）**不是**通用的資料集統計行為，如果烘焙後的 delta 是以
  rot6d 編碼（確實是），仍必須事後套用這個夾限。觀測端統計量也依然需要，做法
  比照（較簡單的）`EEAbs` 式觀測處理。
- **雙重轉換風險（已標示，必須防範）：** 訓練時，任何被登記為 `ee_delta` 的
  transform，都**不得**再對動作重複套用任何相對化——`action` 已經是烘焙好的
  delta，必須原封不動通過，正如 `EEAbsTransform` 目前對自己的 action 欄位所做的
  處理一樣。
- 「完全不註冊任何 transform、直接搭 `joint_abs` 那條無 Transform 路徑」這個做法，
  只有在 `observation.state` **也**被烘焙成硬碟上的 10n rot6d 時才可行（超出本次
  決定的範圍）——不採用此做法，因為那會破壞刻意與 `native` 絕對觀測慣例對齊的
  設計，換來的簡化也很有限。

**GT-replay 的影響（第 3 項）：** 現在要驗證的是「硬碟上烘焙欄位對不對」——從
原始絕對姿態序列，透過同一套 `ee_delta_forward` 數學重新計算出預期的 delta，
並與 mcap_converter 實際寫入的內容比對，而不是透過即時呼叫 Transform 來做
round-trip。

命名：仍使用 `action_type="ee_delta"`，並依「術語」章節的規定，一律只使用
「delta」（converter 端的旗標 `ee_action_encoding` 與訓練端的 action_type 是
兩個不同、各自獨立命名的旋鈕——保持兩者區分，不要混為一談）。排序備註不變：
既有 n-0 程式碼的改名，仍應在本項工作之前或同時完成，這樣 converter 新增的
delta 計算呼叫點，才不會與含糊的 `ee_rel_*` 名稱並列出現。

目標架構：**Diffusion**（已定案）——checkpoint 的 `chunk_size`／
`n_action_steps`／`n_obs_steps` 其餘部分應盡量比照已診斷案例的 checkpoint
（horizon=16, n_action_steps=8, n_obs_steps=2），作為最貼近的對照組，除非訓練
結果顯示應另作調整。

### 第 2 項 — 推論端重新設計（解耦的 delta-mode 發布迴圈）
在動作佇列中儲存模型輸出的**delta**（而非事先還原好的絕對值）。在 delta-mode
發布迴圈中（依 control_freq 頻率，與推論頻率各自獨立）：每個 tick 都讀取最新的
真實觀測，並即時計算 `absolute_target = obs_pose ∘ delta_k`——精確地說是
`absolute_target_xyz = obs_xyz + delta_xyz`、`R_absolute_target = R_delta @
R_obs`（見上方「旋轉數學（已定案）」）——然後再發布。這與 robosuite 每個執行步驟
都重新錨定的做法一致，讓完整的多步 chunk 可以在 checkpoint 訓練時所用的
n_action_steps 下以開迴路方式執行。
**只有在以下條件成立時，等價性才成立（已標示）：** (a) 用來錨定的觀測值是量測到的
目前姿態，而不是指令值（否則會變成推算式的死算，dead-reckoning）；(b) delta
本身確實是單步的（第 1 項）——這兩者是耦合的，需一併驗證；(c) 真實致動的延遲
只會造成有界的滯後（trailing lag），而不是發散——這一點必須被「量測」出來。
這個模式完全不會用到固定錨點的 n-0 路徑（那條路徑——`_delta_ref_state`／
`ee_rel_restore_chunk`，提議改名為 `_relative_anchor_state`／
`ee_relative_restore_chunk`——會繼續保留給既有的 `ee_relative` action_type
使用，不做任何改動，作為對照組）。

### 第 3 項 — OpenArm GT-replay 工具（全新、不依賴模型）——第一道關卡，門檻不變
一個永久可重用的 CLI：載入一個已轉換的 delta-flow 資料集（或原始參考 session 的
GT 姿態），對 GT 絕對姿態執行「新的正向轉換 → 新的反向／發布組合」round-trip，
完全不使用模型，確認能還原到約機器精度等級。重用
`anvil_shared.ee_transform` ＋ `anvil_eval.dataset.EvaluationDataset`，並延伸
scratchpad 中的雛型（`ee_diag/task4_roundtrip.py`）。這是每一個 delta-flow
資料集在投入訓練運算資源之前，都必須先通過的關卡。這不是 LIBERO
bench_spec/gating 的移植版本。**這道門檻不會因為 10-episode 的煙霧測試而放寬**——
它驗證的是轉換數學本身是否正確，與資料集大小或模型品質無關。

### 第 4 項 — 轉換參考 session 並訓練（10-episode 煙霧測試規模——見下方設定）
透過 `mcap-convert`，使用新的 `ee_action_encoding: "delta"` 設定旗標（依上方
重新設計的第 1 項，於轉換時烘焙逐幀 delta），將使用者提供的 10-episode 原始
session 轉換為 delta-flow EE 資料集——需要新增／修改一份 EE 設定檔，既有的
`ee_abs`／`ee_relative` 設定檔不動。先通過第 3 項 GT-replay 關卡。接著透過
`anvil-trainer --action-type=ee_delta` 訓練 **Diffusion**，規模是**煙霧測試
規模，而非收斂規模**——確切旗標見下方「10-episode 煙霧測試設定」。

### 第 5 項 — 離線評估——本次門檻降低
在 checkpoint 上執行 `anvil-eval`（為新的 action_type 新增反向分支）。**本次
通過門檻＝「能跑起來、有輸出」，而不是「通過 PASS/FAIL 門檻」。** 在 n=10 的
規模下，`anvil-eval` 的位置／方向／夾爪指標出現 FAIL 是**預期中的結果，不是
問題**——這個規模下模型準確度差是正常的。真正該留意的失敗訊號是：崩潰、輸出中
出現 NaN/inf，或輸出檔案缺失／為空。可選擇性地額外用 `anvil-eval-ros` 做
MCAP 重播，作為另一個「能跑起來不崩潰」的檢查，門檻同樣降低。

### 第 6 項 — 附監控的真實硬體推論——本次無法執行
這個 session 沒有連接真實 OpenArm。「無法執行時該回報給 Patrick 的內容」見下方
「第 6 項交接檢查清單」。等到真的可以執行時，成功標準＝依診斷方法檢查：
觀測 vs. control_cmd 的抖動比例是否回到約 1.0（相對於失敗案例中的 1.5–3.7 倍），
以及是否出現系統性追蹤延遲（觀測 vs. 指令的相位偏移）——這是發布迴圈可能引入的
新失敗模式。

### 第 2b 項（新增、已確認納入範圍）— 假硬體 EE 擴充
已調查：現有的假硬體系統（`docker-compose.fake-hardware.yml` ＋
`MockControllerNode`、`.../test/fake_hardware/fake_hardware_node.py`）**目前只有
關節空間**——發布的是隨機（非依指令連動的）`/joint_states`，完全沒有 EE 主題，
也不曾碰過 `CommandedEEPose`。不過它的架構確實很有價值：在多容器環境下使用真正的
CycloneDDS，`MultiThreadedExecutor` 搭配各自獨立的
`MutuallyExclusiveCallbackGroup`，觀測（`_obs_timer`）與發布（`_publish_timer`）
使用各自獨立的計時器（inference_node.py:115-127）——這正是第 2 項所引入的解耦
計時器模式。這是離線單元測試與真實硬體之間，一個非常有用的中間層，但要讓它真正
可用，需要**新的模擬邏輯，而不只是設定調整**：
1. 在模擬節點上新增一個 `CommandedEEPose` 發布者，發布到 `/ee_pose_<arm>`
   （訊息串接）。
2. 新增一個 `/commanded_ee_<arm>` 訂閱者，將收到的指令**回聲／積分回去，作為下一次
   發布的觀測值**（`next_ee_pose ≈ last_received_command`）——這是為了實際演練
   delta-mode 的自我修正機制所必須的；只是靜態或隨機發布 EE 姿態，只能驗證主題
   接線是否正確，但**無法**驗證 `obs_pose ∘ delta` 回饋迴圈本身。
**不論如何都明確排除在範圍之外：** 真實的致動動態、實際的速度／延遲限制、感測器
延遲——這個模擬擴充（若真的建置）只驗證軟體端的時序／組合正確性，永遠無法驗證
物理行為。即使做了這個擴充，這一點仍然成立；它不能取代第 6 項。
**範圍決策：本次確認納入範圍。** 這是真正的新程式碼（一個不大的迷你功能——在
模擬節點上新增一個主題發布者 ＋ 一個回聲／積分訂閱者），超出原本六項計畫之外，
但考量到它確實是一個有意義的中間層，且第 6 項本次無法執行，因此明確核准納入。

## 排序（有前提條件）
第 1 項 → 由第 3 項驗證（GT-replay）→ 第 4 項（轉換＋訓練）→ 第 5 項（離線評估）
→ 第 6 項（真實硬體）。第 2 項與第 1 項同步建置（兩者耦合），由第 3 項
（數學正確性）與第 6 項（硬體抖動＋延遲）驗證。**絕對不可**在 GT-replay（第 3 項）
通過之前執行硬體（第 6 項）；**絕對不可**在第 1 項通過 GT-replay 確認之前開始
訓練（第 4 項）。

## 尚待處理的飛行前檢查（非設計決策）
確認 quest_teleop_controller 在自主推論期間會被一併啟動（它與 teleop 使用的是
同一個節點／同一段程式碼路徑，如上文所確認，因此目前部署狀態下應該本就成立——
視為煙霧測試層級的檢查，而非未解決的問題），並確認其擁有權閘門
（`quest_teleop_controller.py:767`）在推論開始時，沒有被其他請求者持有
（例如正在進行 rehoming 操作時）。

## 術語定案（稽核＋改名提案——僅為規劃，尚未實際改名）

**標準用語（往後在程式碼、設定、文件中一律使用）：**
- **Absolute（絕對）** — 原始絕對姿態，從不做相對化。
- **Delta (n-(n-1))（差分）** — 每一步的目標都相對於「緊接在前」*被觀測到*的那一幀；
  逐幀錨點，每一步都重新推導。這是第 1 項引入的新機制。
- **Relative (n-0)（相對）** — 每一步的目標都相對於「chunk 開始時」的觀測值；
  整個 horizon 共用同一個固定錨點。這是正式環境既有的 `ee_rel` 目前的實作方式
  （已診斷出的根因）——「相對（relative）」不再泛指「任何非絕對的表示法」，
  而是特指這一種機制。

**分支位置修正（本次稽核中發現，不是命名問題，但很重要）：** 實際的
`research/libero_ee/` 文件 ＋ `packages/anvil_sim/.../studies/libero_ee/`
程式碼，**並不**存在於目前 repo 中的 `patrick/sim-valid-dev` 分支——它們是在
`research/add-maniskills-env-and-test` @ 660934d 上找到的（worktree：
`.worktrees/add-maniskills-env-and-test`）。值得注意的是，660934d 正是本 session
稍早看到 `origin/sim-valid-dev` 所指向的同一個 commit——這個參照似乎在本 session
期間移動過（很可能是這次對話之外的其他並行程序造成的；`sim-valid-dev`
這個 worktree 也在本 session 期間被另外觀察到切換到了 `main`）。**建議向 Patrick
確認目前哪個 ref／分支才是這段歷史的權威來源**後再進一步引用；在此之前暫時以
`research/add-maniskills-env-and-test` 作為工作依據。

### 正式環境稽核摘要（`implement-ee-space` worktree）
範圍內約有 340 處「rel」／「relative」的出現位置，**目前全部都是 Relative(n-0)**——
Delta (n-(n-1)) 目前尚未存在於程式碼中，只存在於
`claude_docs/ee-delta-flow-plan.md` 這份文件裡。

**公開介面（必須維持向後相容，不可悄悄破壞）：**
1. `action_type` 字串值 `"ee_rel"`——CLI 旗標值、`_VALID_ACTION_TYPES`
   （config.py:71）、所有 `--help`／提示文字（train.py:156-159、
   anvil_eval_ros/cli.py:600）。
2. 已持久化的 `anvil_config.json` 欄位 `"action_type": "ee_rel"` ／
   `"is_ee_rel": true`——**橫跨 5 個模型目錄、共 31 個硬碟上的 checkpoint 檔案**
   （`ee_rel_v1`..`v4` ＋ `diffusion_20260702_145619`，也就是造成已診斷失敗案例的
   那個 checkpoint）。任何改名都必須永久保留讀取舊有 `"ee_rel"` 字串的能力，
   或提供明確的遷移機制。
3. `configs/lerobot_control/inference_ee.yaml` ＋ 煙霧測試設定檔 YAML 中的
   註解／數值。

**僅限內部使用（可自由改名，無相容性疑慮）：** `EERelTransform`、
`ee_rel_forward` / `ee_rel_inverse` / `ee_obs_rel_forward`
（anvil_shared/ee_transform.py）、`ee_rel_restore_chunk`（ee_runtime.py）、
`_compute_ee_rel_stats`、`is_ee_rel` 屬性、各種日誌標籤（`[ee_rel]`、
`[ee_rel_stats]`）以及文件中的說明文字（docs/training.md、
docs/ee_space_report[.zh-TW].md、docs/relative_ee_failure_analysis.md、
本文件、診斷報告）。

**必須解決的關鍵衝突（在第 1 項使用「delta」之前）：** 正式環境目前已經普遍使用
「delta」來指稱 **n-0 機制中的 SE(3) 偏移向量**——`_delta_ref_state`
（固定的 chunk 錨點）、`_ee_rel_action_for_delta`、`n_delta_steps`、
「delta restore」相關註解（inference_node.py:780,786）。這些都**不是**新的
Delta(n-(n-1)) 概念。`action_delta_indices` 很可能是 lerobot 上游自己的欄位名稱
（需要快速確認其所有權，再決定是否要改動——與 LIBERO 端 robosuite 的
`control_mode="relative"` 屬於同一類，是第三方 API，不由我們決定改名）。

### LIBERO 稽核摘要（`research/add-maniskills-env-and-test` worktree）
LIBERO 研究中存在兩條互相獨立的軸線：目標**表示法（representation）**（我們的
標準分類）vs. **傳遞方式（delivery）**（`deliver="absolute"/"relative"/
"relative_converted"`——重建出的目標要如何餵給 robosuite）。大多數命名上的
不一致，都是因為把這兩條軸線混在一起造成的。

| 條件 | 實際機制 | 標準分類 | 命名是否正確？ |
|---|---|---|---|
| native / native_rot6d / native_hand | 逐步記錄的 delta，以即時觀測為錨點 | **Delta (n-(n-1))** | ✅ 是 |
| `native_n0` | 目標在轉換時就逐幀相對化（評估時強制 `per_frame_anchor=True`）；退化到近似 native | **Delta (n-(n-1))**，退化版本 | ❌ **命名有誤**——「n0」錯誤地暗示是 Relative(n-0)。建議改名為例如 `native_perframe_baked`。 |
| `goal-world-n0` / `goal-hand-n0` | 絕對目標相對於 **chunk 起始**錨點做相對化（`per_frame_anchor=False`） | **Relative (n-0)** | ✅ 是——「n0」正確表示 chunk 起始 |
| `goal-abs` / `native_abs` | 形式上的 `state+Δ`，未經縮放，沒有一致的物理單位 | Absolute（形式上／退化版本） | ⚠️ 部分正確——名稱本身沒有誇大，但在每處引用時都要註明此警示 |
| `native_ctrlgoal` | 真正物理意義上的絕對控制器目標 | **Absolute** | ✅ 是 |
| `native_ctrlgoal_relconv` | 同一個絕對目標；只有*傳遞方式*是 `relative_converted` | Absolute（目標本身）；`relconv` 是傳遞方式，不是表示法 | ✅ 保留，但需標註此處的「rel」與我們的錨點分類不是同一件事 |
| `afo_abs_h{1,5,10}` | 真正觀測到的未來絕對姿態（OAA） | **Absolute** | ✅ 是 |
| `afo_relative` | 與 `afo_abs_h1` **完全相同**的絕對目標；只有*傳遞方式*是 `relative_converted`——完全沒有相對於任何錨點做相對化 | **Absolute**（由觀測推導） | ❌ **命名有誤**——這裡的「relative」指的是傳遞方式，與新定義的意涵衝突。建議改名為例如 `afo_abs_relconv`（與 `native_ctrlgoal_relconv` 對應）。 |
| `replay_adapter.py` 的 `"direct"` provider | 屬於重播機制層級的標籤，橫跨多個標準分類 | 不適用（正交的軸線） | ✅ 不需改名——不要硬把它塞進這套分類裡 |
| `rel_world`/`rel_hand`（provider 層級） | 在 provider 標籤層級上正確對應 Relative(n-0) | Relative (n-0) | ✅ 是，但底層 `ZeroCalActionProcessorStep` 的 **mode** 字串是超載使用的（也驅動了 `native_n0` 強制逐幀的 Delta）——建議加上消歧義的註解，不一定需要改名 |

### 改名方案提案（尚未執行——僅為提案）
- **正式環境，公開的 `action_type` 字串：`"ee_rel"` → `"ee_relative"`**
  （已定案——採簡短形式；在新的標準分類下「relative」已無歧義，因此不需要額外的
  `_n0` 後綴）。載入端必須永久接受舊有 `"ee_rel"` 字串，作為所有 31 個既有
  checkpoint 的別名。
- **正式環境，內部使用：** `EERelTransform`→`EERelativeTransform`；
  `ee_rel_forward`/`ee_rel_inverse`/`ee_obs_rel_forward`→
  `ee_relative_forward`/`_inverse`/`ee_obs_relative_forward`；
  `ee_rel_restore_chunk`→`ee_relative_restore_chunk`；
  `_compute_ee_rel_stats`→`_compute_ee_relative_stats`；
  `is_ee_rel`→`is_ee_relative`。（類別／函式名稱仍可透過每處的標準用語
  docstring／註解隱含表達「n-0」的語意——一旦「relative」本身已無歧義，明確的
  `_n0` 後綴就被判定為非必要，與上述公開 token 的決策一致。）
- **正式環境，解決「delta」的命名衝突：** `_delta_ref_state`→
  `_relative_anchor_state`；`_ee_rel_action_for_delta`→
  `_ee_relative_action_for_offset`；`n_delta_steps`→`n_offset_steps`；
  將「delta restore」相關註解改寫為「n-0 restore」／「chunk-anchor restore」。
- **第 1 項新功能——一律只用「delta」，絕不使用「rel」：**
  `action_type="ee_delta"`（已規劃）；`EEDeltaTransform`；
  `_compute_ee_delta_stats`。**設計備註（修正——見上方「旋轉數學（已定案）」）：**
  `ee_delta_forward`/`ee_delta_inverse` **不是** `ee_rel_forward`/
  `ee_rel_inverse` 逐樣本錨點分支的薄包裝——該分支本身與錨點無關（可以重用其
  *傳遞錨點的模式*），但其平移與旋轉都是寫死的本體座標系組合，而新類型的平移與
  旋轉都需要世界座標系組合（已對照 robosuite 1.4.0 原始碼驗證）。新函式必須
  依修正後的公式重新撰寫；只有底層的旋轉矩陣／rot6d primitives 是共用的。
- **LIBERO：** `native_n0`→`native_perframe_baked`（或類似名稱）改名；
  `afo_relative`→`afo_abs_relconv`。傳遞方式軸線的名稱維持不動
  （`relative_converted`、robosuite 的 `control_mode="relative"`、provider 的
  `rel_world`/`rel_hand`）——這是不同的軸線，且 robosuite 的情況屬於第三方
  API——但要在超載使用的 `ZeroCalActionProcessorStep` mode 字串上加註消歧義的
  說明。
- **需要更新的文件（僅文字內容，用語不變）：**
  `research/libero_ee/{stage1-closeout,report}.md`
  （add-maniskills-env-and-test worktree）；本 worktree 的
  `claude_docs/ee-space-libero-vs-production-diagnosis.md` 與
  `claude_docs/ee-delta-flow-plan.md`；`docs/training.md`、
  `docs/ee_space_report[.zh-TW].md`、`docs/relative_ee_failure_analysis.md`。

本節僅為稽核與提案——尚未執行任何改名。若獲核准執行，應作為獨立、有前提條件的
一個步驟，很可能安排在第 1 項之前或同時進行，因為第 1 項的新程式碼不應該一出生
就落入上述的「delta」命名衝突之中。

## 本次 session 的執行範圍確認

依明確指示：**所有實作工作僅限於 `patrick/implement-ee-space`**
（worktree `.worktrees/implement-ee-space`，直接在該分支上工作——不建立新的
巢狀 worktree／分支，這是 Patrick 對「每次改動都建立獨立 worktree」這個預設習慣的
明確例外指示）。`research/add-maniskills-env-and-test` 與
`patrick/sim-valid-dev` 在本次工作中僅作為參考——只讀取事實，絕不修改。

**環境檢查（本次 session）：** GPU 可用（RTX 4090）→ 第 4 項（訓練）在此環境中
是可行的。此 shell 的 PATH 中找不到 `ros2` → 第 6 項（真實硬體部署）**無法從此
沙盒環境執行**，需要 Patrick 實際的機器人主機。第 5 項（離線評估）在此環境中
是可行的（不需要硬體）。

**本次 session 於 worktree 中觀察到的既有未提交狀態（並非本 session 造成）：**
`scripts/run_inference.sh` 在本 session 進行任何修改之前，就已顯示為已修改狀態
（尚未 stage／commit）。不要悄悄地把它一併提交覆蓋掉；應檢查其 diff 內容，
若無法確定是否為刻意進行中的工作，應提出來，而不是自行假設。

**本次執行順序提案（有前提條件，符合計畫本身「絕不可在第 1 項通過
GT-replay 確認前訓練」的排序規則）：**
1. 執行術語改名（正式環境端：`ee_rel`→`ee_relative` 家族，解決「delta」
   命名衝突）——這是前提條件，如此第 1 項的新程式碼才不會一出生就與含糊的名稱
   並列。**要以行為驗證，而不只是靠推理**：載入至少一個真實既有 checkpoint
   （例如 `diffusion_20260702_145619`，也就是已診斷出問題的那一個），走過改名後
   的程式碼路徑，確認舊有的 `"ee_rel"` 字串仍能正確對應到
   `EERelativeTransform`／`"ee_relative"` 的行為——要有具體證據證明改名沒有
   悄悄破壞既有 checkpoint，而不只是「diff 看起來沒問題」。
2. 第 1 項——依重新設計後分成兩部分：(a) `mcap_converter` 的改動（新增
   `ee_action_encoding: "delta"` 設定旗標 ＋ `prev_state` 管線串接 ＋ 第一幀
   自我錨定慣例，使用新撰寫的世界座標系 `ee_delta_forward`，依上方「旋轉數學
   （已定案）」）——在轉換時將 delta 烘焙到硬碟上；(b) 訓練端精簡許多的部分——
   一個只處理觀測的 Transform（模仿 `EEAbsTransform`，action 直通）＋
   `_compute_ee_delta_stats`（從靜態欄位算出 mean/std/min/max ＋ rot6d
   identity 夾限）。新的 stats 方法**必須複製既有的 epsilon 下限模式**（見下方）
   ——這是必要項目，不是可有可無的加固。
3. 第 2 項——推論端的解耦 delta-mode 發布迴圈，使用世界座標系的反向公式
   （`absolute_target_xyz = obs_xyz + delta_xyz`、`R_absolute_target =
   R_delta @ R_obs`）。
4. 第 2b 項——假硬體 EE 擴充（已確認納入範圍）。
5. 第 3 項——GT-replay 工具（不依賴模型的 round-trip），門檻嚴格。**關卡：
   必須在第 4 項之前通過。**
6. 第 4 項——轉換 10-episode session 並以煙霧測試規模訓練（此環境可行，GPU
   可用）。確切設定見下方。
7. 第 5 項——離線評估（`anvil-eval`），門檻降低（能跑起來且不崩潰／不出現
   NaN 即可，PASS/FAIL 結果不論）。
8. 第 6 項——真實硬體部署：**本次 session 無法執行**——改為產出下方的交接
   檢查清單。

### 10-episode 煙霧測試設定（已調查，具體內容）
建議：**`--split-ratio=8,1,1 --steps=10 --save_freq=10 --batch_size=1
--num_workers=0 --eval_freq=0 --log_freq=5 --action-type=ee_delta`**——這是
比照 `tests/smoke/scripts/pipeline_smoke_test.py` 既有的驗證過先例（5-episode
fixtures，`3,1,1` 切分比例，相同的 steps/save_freq/batch/worker/eval 模式），
放大到 10 個 episodes，並採用預設形狀的 `8,1,1` 比例（→ 8 訓練 / 1 驗證 / 1
測試）。**對原始框架的修正：** 曾考慮過 `--split-ratio=1,0,0`（全部 10 個都當
訓練集，關閉驗證／測試），但這樣做對本次目的來說其實是明確錯誤的——
`apply_val_loss_patch`（patches.py:585-588）在驗證＋測試比例 ≤0 時會提前
return，這會導致完全不安裝 `patched_make_dataset`，意味著**新的統計量計算方法
根本不會被執行**，違背了這次煙霧測試原本就是要驗證這件事的目的。請改用
`8,1,1`（或任何驗證比例 >0 的比例）——**已確認、已定案（接受此項修正）。**
`--steps=10 --save_freq=10` 保證不論規模大小，都恰好會存一個 checkpoint，
因為 lerobot 自己的存檔條件無條件包含 `step == cfg.steps`
（lerobot_train.py:447,469）——確認這不是收斂規模的訓練，純粹是存檔路徑的
煙霧測試。

**統計量健全性（已調查，具體內容）：** 既有的 `_compute_ee_rel_stats` /
`_compute_ee_abs_stats` 已經在每個相關位置透過
`np.where(std < 1e-6, 1e-6, std)` 把 std 下限設在 `1e-6`
（patches.py:363,419,488-490,512）——**這個防護機制是必要的，新的單步統計量
方法必須逐字複製這個做法**；在僅有 10 個 episodes 的情況下，夾爪／旋轉維度
接近常數是很實際會發生的零變異數情況。min/max 在 anvil 的程式碼中，除了 rot6d
±1 強制夾限之外並沒有加防護，且夾爪的統計量是從絕對值還原來的——在 n=10 的規模
下，位置維度出現退化的零範圍是可能發生的，但這**不需要**由 anvil 再解一次：
lerobot 自己內建的 normalizer（`normalize_processor.py:349-355,372-374,
389-391`）在 MEAN_STD 與 MIN_MAX 兩種模式下，都已獨立透過
`torch.where(denom==0, eps, denom)` 防護了零範圍的情況，因此不論哪種情況都不會
有崩潰／NaN 的路徑——最壞的情況只是某個退化維度悄悄地被映射成常數。

**本次煙霧測試的驗證要求（需要具體證據，而非籠統的判斷）：**
(a) 執行完之後，回報 `_compute_ee_delta_stats` 針對這個特定 10-episode 資料集
實際記錄下來的逐維度 mean/std/min/max 數值——明確說明哪些維度（如果有的話）
出現接近零的 std，或者 min/max 範圍出現塌縮。單單「沒有發生崩潰」並不足以
作為證據。
(b) 整個統計量方法都被包在一個寬鬆的 `try/except` 裡，失敗時會悄悄退回使用原始
資料集統計量（patches.py:443-445,534-536）——這表示不論新的統計量路徑真的有
執行，或是悄悄退回了後備方案，「訓練順利完成、沒有出錯」看起來都會一模一樣。
如果還沒有的話，請新增一行明確的日誌，確認 `_compute_ee_delta_stats`
確實執行完畢，**而且**它的輸出確實是被真正注入到
`train_dataset.meta.stats["action"]` 的那一個（而不是後備方案）——然後從實際
執行結果中引用這一行日誌作為證據，而不只是「這次執行完成了」。

### 第 6 項交接檢查清單（骨架——待第 4 項產出 checkpoint 後填入具體數值）
等到真實硬體部署變得可行時，需向 Patrick 回報：
- 已訓練 checkpoint 的路徑（位於
  `model_zoo/ee-space/<dataset>/<run>/checkpoints/<step>/` 之下）。
- 該使用哪一份推論設定檔（`configs/lerobot_control/inference_ee.yaml`，
  確認 `ee_command_topic`/`ee_obs_topic` 與新的 delta-mode 發布迴圈所預期的
  一致）。
- 飛行前檢查：(1) 確認 `quest_teleop_controller` 會一併啟動，且其擁有權閘門
  沒有被其他地方持有（上文已標示過）；(2) 確認控制器的主題命名
  （`/commanded_ee_<arm>`、`/ee_pose_<arm>`）與新發布迴圈程式碼所訂閱／發布的
  內容完全一致——這是依 `inference_ee.yaml` 自身標頭註解所述、屬於
  anvil-workcell 端的相依性，無法單從本 worktree 驗證；(3) 下方標示的跨 repo
  資料過舊風險——在依賴「持續發布」作為唯一防止指令過舊被伺服的機制之前，
  先確認已意識到這個風險。

## 已標示的跨 repo 風險（非本計畫引入，但與第 2 項／第 6 項相關）
在 `anvil-workcell/TODO_commanded_ee_stale_target.md` 中發現的問題。控制器對
`/commanded_ee_<arm>` 的消費方式——正是我們新的 delta-mode 發布迴圈所寫入的
那個主題——**完全沒有資料過舊／時間戳檢查**：
`quest_teleop_controller.py:196-206` 會快取最後一次收到的指令目標，且永遠不會
重置它；`absolute_control_modality.py:218-219` 使用該值時只做了存在性檢查。
若我們的發布迴圈停止運作（崩潰、卡住、網路短暫中斷），手臂會持續無限期地伺服到
最後一次發布的目標，而不是安全地失效停止。這是 `anvil-workcell` 既有的臭蟲，
不在本計畫六項範圍內（跨 repo 問題），但：(a) 這表示我們的新迴圈在設計時，
不應在沒有考量這個「失效即開啟（fail-open）」行為的前提下，再引入額外的卡住
模式；(b) 這值得提出來讓 `anvil-workcell` 的負責人知道，這是一個與 delta-flow
工作本身無關、真實存在的硬體安全缺口。
