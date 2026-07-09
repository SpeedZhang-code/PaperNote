# HSTGCNT:分層時空圖卷積與 Transformer 網絡 論文筆記
### 發表於 IEEE Transactions on Intelligent Transportation Systems

> 本檔案已將所有公式轉換為 GitHub 原生支援的 LaTeX 語法(`$...$` 行內、`$$...$$` 區塊),並移除可能在 GitHub 上顯示異常的巢狀 `\text{}` 標號寫法,改以獨立文字標示公式編號。

---

## 📌 論文核心摘要與主要貢獻

**解決痛點:**
1. 傳統基於圖卷積網絡(GCN)的預測模型,難以同時完美捕捉交通數據中的「短期(Drastic)」與「長期(Long-term)」時間關聯
2. GCN 模型在層數加深時,存在嚴重的過平滑問題(Over-smoothing),導致節點特徵同質化、預測退化

**核心創新:** 採用雙路並行分層網絡架構。上層設計長期時間 Transformer 網絡(LTT)抓取全局長期規律;下層設計時空圖卷積網絡(STGC)透過 1D 卷積與圖卷積串聯抓取短期局部時空特徵。中間透過交叉注意力融合模組(LSTIF)層層動態融合,並藉此注入外部長效信息以緩解 GCN 的過平滑困境。

**主要貢獻:**
- 提出並行提取長/短期時間依賴的交通預測新架構
- 改良位置編碼,加入「週期性時間步長」以適應交通流的規律特徵
- 提出動態交叉注意力特徵融合機制,在數學上證明了該機制能有效抑制圖卷積的過平滑現象

---

## 🧮 核心公式詳細解析

### 1. 空間圖卷積(GCN)的標準形式

本模型底層的空間關聯依然建立在對非歐幾里得道路網絡的圖卷積之上:

$$
f(\mathbf{Z}^{(l)}, \mathbf{A}) = \sigma\left(\hat{\mathbf{D}}^{-\frac{1}{2}} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-\frac{1}{2}} \mathbf{Z}^{(l-1)} \mathbf{W}^{(l)}\right)
$$

**(公式 1)**

**解析:**
- $\hat{\mathbf{A}} = \mathbf{A} + \mathbf{I}$:引入自環的地鐵/道路鄰接矩陣
- $\hat{\mathbf{D}}$:$\hat{\mathbf{A}}$ 的對角節點度矩陣
- $\hat{\mathbf{D}}^{-\frac{1}{2}} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-\frac{1}{2}}$:對稱歸一化拉普拉斯矩陣,用於聚合鄰居路段的客流特徵
- $\mathbf{Z}^{(l-1)}$ 與 $\mathbf{W}^{(l)}$:分別是第 $l$ 層的特徵輸入與可學習權重矩陣

---

### 2. 週期性改良型時間位置編碼(Temporal Position Embedding)

傳統 Transformer 的位置編碼由於設定的週期(分母)過長,無法表現交通客流數據中強烈的日/週規律性。因此,本論文提出了自定義週期(Pre-set period)的位置編碼:

$$
\text{PE}(K) = \sin\left(\frac{2\pi K}{\text{period}}\right)
$$

**(公式 4)**

**解析:**
- $K$:當前序列中的絕對時間步(如當天第幾分鐘)
- $\text{period}$:預設的交通變化規律週期
- **物理意義**:這使得編碼出來的特徵矩陣具有極高的判別度,能夠讓模型直接識別出「週一早高峰」或「週五晚高峰」的週期規律

隨後將原始客流數據 $\mathbf{X}$ 與該編碼 $\text{PE}$ 進行矩陣拼接,作為 LTT 網絡的總輸入:

$$
\mathbf{X}^* = \text{Concat}(\mathbf{X}, \text{PE})
$$

**(公式 5)**

---

### 3. 長期時間 Transformer 網絡(LTT)編碼器

利用 Self-Attention 機制,LTT 能夠實現雙向、全局並行的長序列時間特徵學習(打破了 RNN 無法並行且易梯度消失的限制)。第 $l$ 層編碼器的抽象表示如下:

$$
\mathbf{H}^{(l)} = \text{LTT}(\mathbf{H}^{(l-1)}) = \text{LN}\left(\text{dropout}\left(\text{FeedForward}(\text{Res}^{(l)})\right) + \text{Res}^{(l)}\right)
$$

**(公式 6)**

其中核心的多頭自注意力機制(Multi-Head Attention)公式為:

$$
\text{head}_m = \text{softmax}\left(\frac{\mathbf{H}^{(l-1)}\mathbf{W}_m^Q \left(\mathbf{H}^{(l-1)}\mathbf{W}_m^K\right)^T}{\sqrt{D_p + D_r}}\right)\mathbf{H}^{(l-1)}\mathbf{W}_m^V
$$

**(公式 8)**

**解析:**
- $\mathbf{W}_m^Q, \mathbf{W}_m^K, \mathbf{W}_m^V$:代表查詢(Query)、鍵(Key)、值(Value)的線性映射權重矩陣
- $\sqrt{D_p + D_r}$:縮放因子,用以防止矩陣點積後的數值過大導致梯度平緩
- **物理意義**:計算序列中任意兩個時間點客流(例如 1 小時前與現在)之間的全局關聯權重

---

### 4. 短期時間卷積(STC)與門控線性單元(GLU)

在底層的 STGC 分支中,為了抓取突發的、短期的客流局部劇烈波動,模型採用了 1D 卷積,並用門控線性單元(GLU)來防止傳統 Tanh 激活函數引發的梯度消失:

$$
(\beta_1, \beta_2) = \text{split}\left(\text{Conv1d}(\mathbf{Z}^{(l-1)})\right)
$$

$$
\text{STC}(\mathbf{Z}^{(l-1)}) = \beta_1 \odot \text{sigmoid}(\beta_2)
$$

**(公式 11)**

**解析:** $\text{Conv1d}$ 沿著時間軸提取特徵後,將其平分為 $\beta_1$ 和 $\beta_2$,透過 Hadamard 積($\odot$ 矩陣點乘)進行激活過濾,使短期特徵梯度能夠非常平穩地進行反向傳播。

---

### 5. 異構多頭交叉注意力融合模組(LSTIF)

這是連接長/短期雙路網絡、解決過平滑問題的最核心設計。它將長期 LTT 特徵映射為 Query,短期 STC 特徵映射為 Key,相乘後作為權重分配給融合特徵 Value:

$$
\mathbf{F}^{(l)} = \text{Fusion}(\mathbf{H}^{(l)}, \mathbf{T}^{(l)}) = \text{Attention}\left((\mathbf{T}^{(l)}, \mathbf{H}^{(l)}) \cdot \mathbf{Y}^{(l)}\right)
$$

**(公式 15)**

其實現多頭機制的內部公式為:

$$
\text{head}_m = \text{SoftMax}\left(\frac{\mathbf{K}_m (\mathbf{Q}_m)^T}{\sqrt{D_1/2}}\right) \cdot \mathbf{V}_m
$$

$$
\text{其中} \quad \mathbf{K}_m = \mathbf{W}_m^T \mathbf{T}^{(l)}, \quad \mathbf{Q}_m = \mathbf{W}_m^H \mathbf{H}^{(l)}, \quad \mathbf{V}_m = \mathbf{W}_m^Y \mathbf{Y}^{(l)}
$$

**(公式 17)**

**解析:**
- $\mathbf{H}^{(l)}$:長期 Transformer 特徵
- $\mathbf{T}^{(l)}$:短期 1D 卷積特徵
- $\mathbf{Y}^{(l)}$:長短期的拼接特徵(Value)
- **物理意義**:在平峰期,模型會自動通過交叉注意力賦予長期規律(LTT)更高的預測權重;在高峰期或有突發事件時,則賦予短期局部波動(STC)更高權重,達成動態分層調節

---

### 6. 多任務總損失函數(Objective Function)

為了確保 Transformer 提取出的特徵既具備客流預測能力,又保留完整的時間規律特徵,本論文採用了多任務聯合學習策略:

$$
\mathcal{L} = \mathcal{L}_{stgc\text{-}mse} + \lambda_1 \mathcal{L}_{stgc\text{-}mae} + \lambda_2 \mathcal{L}_{trans\text{-}res} + \lambda_3 \mathcal{L}_{trans\text{-}fore}
$$

**(公式 18)**

**解析:**
- $\mathcal{L}_{stgc\text{-}mse}$ / $\mathcal{L}_{stgc\text{-}mae}$:底層時空網絡的預測損失,MSE 對極端異常值敏感,MAE 適合穩定期的客流
- $\mathcal{L}_{trans\text{-}fore}$:LTT 自帶的長期預測 Loss
- $\mathcal{L}_{trans\text{-}res}$:LTT 解碼器的流量重構損失(Reconstruction Loss),確保時間規律不遺失

---

## 📊 實驗結果與消融研究結論

論文在三個公開數據集上進行了驗證:PeMS-BAY(高速公路車速)、PeMSD7(M)(加州公路流量)以及北京地鐵客流(Beijing Metro)。

### 1. 實驗結果表現(參見論文 Table II & III)

- 在與主流模型(如 STGCN, DCRNN, Graph WaveNet)對比中,HSTGCNT 在中長期預測(45~60分鐘)表現出壓倒性的優勢
- 與目前最主流、極難超越的 Graph WaveNet(GWN)相比,本模型在 PeMS-BAY 數據集上使 MAE 降低了 6%,RMSE 降低了 2.3%
- 在北京地鐵數據集上的表現尤其突出。原因在於地鐵路網結構相對簡單(非換乘站的節點度僅為 2),純空間特徵不如時間特徵複雜,而本模型強大的 LTT(長期時間網絡)恰好彌補了其他時空模型忽略長期時間規律的缺陷

### 2. 消融實驗與過平滑視覺化(Ablation Studies & Visualization)

為了驗證模型如何解決 GCN 的過平滑病灶,論文設計了變體並進行了矩陣視覺化(參見論文 Fig. 4):

- **HSTGCNT-wLTT(拔除長期網絡)**:隨著 GCN 層數疊加,其重構出的鄰接矩陣(Fig. 4c)變得極度模糊且均勻,證明數據進入了過平滑狀態,失去了路段的個體特異性
- **HSTGCNT(完整模型)**:即使疊加了多層網絡(Fig. 4e),其矩陣圖案依然紋理清晰、層次分明

**結論證明:** 利用 LSTIF 融合模組,將 Transformer 捕獲的高階長效特徵層層注入到 GCN 中,能夠為圖卷積源源不斷地補給外源特徵信息,從數學和視覺上完美破解了圖神經網絡「無法疊加深層」的過平滑死結。
