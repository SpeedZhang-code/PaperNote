# ARConv-GCN:地鐵短途客流量預測 論文筆記
### 發表於 IET Intelligent Transport Systems

> 本檔案已將所有公式轉換為 GitHub 原生支援的 LaTeX 語法(`$...$` 行內、`$$...$$` 區塊),並移除可能在 GitHub 上顯示異常的巢狀 `\text{}` 標號寫法,改以獨立文字標示公式編號。

---

## 📌 論文核心摘要與主要貢獻

**解決痛點:**
1. 現有方法常將空間與時間模式分開建模,忽略了兩者內部的交互作用。
2. 圖卷積網絡(GCN)通常只能疊加 1 到 4 層,過深會導致性能下降(過平滑問題),難以提取深層空間特徵。
3. 傳統 2D CNN 無法高效同時提取高維的時空同步特徵。

**核心創新:** 結合圖卷積網絡(GCN)與改良型 3D 卷積神經網絡(ARConv)。利用 GCN 捕捉地鐵站網絡拓撲的空間關聯,再透過包含「殘差結構」與「注意力機制」的 3D CNN,進行深度的時空特徵融合。

**主要貢獻:**
- **ARConv-GCN 架構**:首次將地鐵站拓撲結構(GCN)與 3DCNN 的多模式時空融合相結合
- **解決深層訓練難題**:引入殘差結構(Residual Structure),使 3D 網絡可以安全加深而不必擔心梯度消失,並結合注意力機制動態調整歷史特徵權重
- **真實數據驗證**:在北京和廈門地鐵數據集上展示了最尖端的預測準確度

---

## 🧮 核心公式詳細解析

### 1. 歷史客流的三種規律模式(特徵矩陣形式)

地鐵客流具有強烈的週期規律。模型將歷史進站(Inflow)與出站(Outflow)數據聚合為三種時空模式矩陣:近期模式(Recent)、日模式(Daily)和週模式(Weekly)。

$$
\mathbf{X}_{N,T}^p = \begin{pmatrix} X_{1,t}^p & X_{1,t-1}^p & X_{1,t-2}^p & \cdots & X_{1,t-m+1}^p \\ X_{2,t}^p & X_{2,t-1}^p & X_{2,t-2}^p & \cdots & X_{2,t-m+1}^p \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ X_{n,t}^p & X_{n,t-1}^p & X_{n,t-2}^p & \cdots & X_{n,t-m+1}^p \end{pmatrix}
$$

**(公式 1)**

**解析:**
- $p \in \{recent, daily, weekly\}$:代表三種客流時間規律模式
- $n$:地鐵站的總數量(矩陣的行數)
- $m$:預測時所採用的歷史時間步長(矩陣的列數)
- $X_{j,k}^p$:代表地鐵站 $j$ 在時間間隔 $k$ 的客流量特徵向量

模型的終極目標是學到一個映射函數 $f$,利用這三種歷史矩陣來預測下一個時間步的客流量 $\mathbf{X}_{n,t+1}$:

$$
\mathbf{X}_{n,t+1} = f(\mathbf{A}; \mathbf{X}_{N,T}^r, \mathbf{X}_{N,T}^d, \mathbf{X}_{N,T}^w)
$$

**(公式 2)**

其中 $\mathbf{A}$ 是地鐵網絡拓撲的鄰接矩陣。

---

### 2. 圖卷積網絡(GCN)的空間拓撲特徵捕捉

傳統方法把地鐵站視為獨立網格,忽略了軌道網絡結構。本論文使用圖卷積來對物理車站之間的相連性進行編碼:

$$
\mathbf{X}^{(l+1)} = \sigma \left( \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{X}^{(l)} \mathbf{W}^{(l)} + b^{(l)} \right)
$$

**(公式 3)**

**解析:**
- $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$:$\mathbf{A}$ 為車站相連的鄰接矩陣,$\mathbf{I}$ 為單位矩陣。加上 $\mathbf{I}$ 代表引入車站「自身客流」的自環(Self-loop)
- $\tilde{\mathbf{D}}$:$\tilde{\mathbf{A}}$ 的對角節點度矩陣(Degree Matrix),用於對圖矩陣進行歸一化
- $\tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}}$:對稱歸一化拉普拉斯矩陣。物理意義是讓模型在聚合地鐵站鄰居客流時,消除因為個別大站(如轉乘站)因連線過多而導致的數值發散,使特徵在圖中穩定傳播
- $\mathbf{X}^{(l)}$ 與 $\mathbf{W}^{(l)}$:分別是第 $l$ 層的輸入特徵矩陣與可學習權重矩陣
- $\sigma$:ReLU 激活函數,用於引入非線性特徵

---

### 3. 進出站流量特徵拼接(Concatenation)

模型分別建立了「進站」和「出站」兩個獨立的 GCN 分支,並將兩者的輸出拼接在一起,作為後續 3D CNN 的輸入:

$$
\mathbf{V} = \mathbf{V}_{inflow}^{l+1} \oplus \mathbf{V}_{outflow}^{l+1}
$$

**(公式 4)**

**解析:** $\oplus$ 代表沿著通道或特定維度軸將兩組張量疊加(Stack)在一起,使得模型能夠同時考慮進站與出站特徵的協同關聯。

---

### 4. 3D 卷積網絡(3D CNN)的客流同步時空提取

2D 卷積只能抓取車站間的純空間特徵,而 3D 卷積透過 3D 濾波器(Filters),能同時在「空間(車站)」與「時間軸」上滑動,達成真正的時空融合:

$$
\mathbf{V}^{l+1} = \text{Conv3D} \circ (\mathbf{K} \cdot \mathbf{V} + \mathbf{\Delta})
$$

**(公式 5)**

其實現特徵圖(Feature map)中特定位置 $(x,y,z)$ 的數值計算公式如下:

$$
v_{ij}^{xyz} = \text{ReLU} \left( b_{ij} + \sum_{m} \sum_{p=0}^{P_{i-1}-1} \sum_{q=0}^{Q_{i-1}-1} \sum_{r=0}^{R_{i-1}-1} w_{ijm}^{pqr} \, v_{(x+p)(y+q)(z+r)}^{(i-1)m} \right)
$$

**(公式 6)**

**解析:**
- $v_{ij}^{xyz}$:第 $i$ 層、第 $j$ 個特徵圖在三維座標 $(x,y,z)$ 上的激發值
- $m$:前一層 $(i-1)$ 的特徵圖索引
- $w_{ijm}^{pqr}$:3D 卷積核(Kernel)在立體相對座標 $(p, q, r)$ 上的權重值
- $(P, Q, R)$:3D 卷積核在三個維度上的立體尺寸(如 $3 \times 3 \times 3$)
- **物理意義**:這一步能把地鐵客流隨著時間在不同相鄰站點之間的「動態擴散(如早高峰人流從住宅區車站流向商業區車站的過程)」完美地捕捉下來

---

### 5. 3D 注意力機制模組(Attention Mechanism)

由於近期、日、週三種模式對未來客流的預測貢獻度不同,模型利用注意力機制為其分配不同的全局權重:

$$
\boldsymbol{\delta}' = \text{Softmax} \left( (\mathbf{R} \circ (\text{Conv3D} \circ v)) \otimes (\mathbf{R} \circ (\text{Conv3D} \circ v)) \right)
$$

**(公式 7)**

$$
\boldsymbol{\delta}'' = \mathbf{R} \circ (\text{Conv3D} \circ v)
$$

**(公式 8)**

$$
\tilde{v} = \text{Conv3D} \circ (\mathbf{R} \circ (w\,\boldsymbol{\delta}' \otimes \boldsymbol{\delta}''))
$$

**(公式 9)**

$$
y = \tilde{v} \oplus v
$$

**(公式 10)**

**解析:**
- $\mathbf{R}$:重塑矩陣形狀(Reshape)操作
- $\otimes$:矩陣相乘(Matrix Multiplication);$\oplus$:矩陣相加(Matrix Addition)
- **運行原理**:公式 7 透過自身特徵矩陣相乘計算出「自注意力權重矩陣 $\boldsymbol{\delta}'$」,代表不同歷史規律特徵之間的關聯權重。公式 9 將此權重施加到重新配置的特徵矩陣 $\boldsymbol{\delta}''$ 上。最後,公式 10 透過矩陣相加與原始特徵相結合,輸出調整過全球權重的最核心特徵

---

### 6. 損失函數(Loss Function)

模型在預測時,採用地鐵客流預測中最主流的均方誤差(MSE)作為損失函數來進行反向傳播優化:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

**(公式 11)**

**解析:** $y_i$ 是 AFC 系統記錄到的真實車站進/出站人數,$\hat{y}_i$ 是模型的預測人數,$N$ 是客流樣本總數。

---

## 📊 實驗結果與消融研究結論

論文使用了兩個真實世界的地鐵 AFC 刷卡數據集進行測試:北京地鐵(MetroBJ,17條線、276個車站)與廈門地鐵(MetroXM,2條線、52個車站),時間間隔均聚合為 10 分鐘。

### 1. 模型表現對比(MetroBJ & MetroXM)

在與傳統統計模型(HA, ARIMA)以及經典深度學習模型(LSTM, 2D CNN, STGCN, DCRNN)的對比中,ARConv-GCN 取得了全面性的顯著提升:

- **在北京地鐵數據集上**:相較於先前最優的基準模型(Conv-GCN),ARConv-GCN 實現了 10.35% 的 RMSE 提升以及 7.43% 的 MAE 降低
- **在廈門地鐵數據集上**:實現了 12.85% 的 RMSE 提升以及 7.99% 的 MAE 降低

### 2. 消融研究(Ablation Study)分析

為了驗證模型內部各改良模組的有效性,論文設計了三種變體進行對比:

- **ARConv-GCN-AR**:拔除注意力機制與殘差模組
- **ARConv-GCN-R**:拔除殘差模組
- **ARConv-GCN-A**:拔除注意力機制

消融結果表明(參見論文 Table 4):

1. 拔除殘差模組後性能下降最為嚴重,這證明了殘差結構在地鐵這種超深層的 3D 卷積網絡中,對於防止梯度消失和提取長距離特徵至關重要
2. 完整版的 ARConv-GCN 表現最好,證明注意力機制精準調整了近期、日、週規律的特徵權重,顯著提高了預測上限
