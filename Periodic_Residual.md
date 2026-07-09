# PRNet:週期性殘差學習網絡 論文筆記
### Periodic Residual Learning for Crowd Flow Forecasting(ACM SIGSPATIAL '22)

> 本檔案已將所有公式轉換為 GitHub 原生支援的 LaTeX 語法(`$...$` 行內、`$$...$$` 區塊),並移除可能在 GitHub 上顯示異常的巢狀 `\text{}` 標號寫法與 `\Vert{}` 等非標準寫法,改以標準 `\left\lVert \cdot \right\rVert` 語法呈現。

---

## 📌 論文核心摘要與主要貢獻

**解決痛點:** 現有時空模型(CNN/RNN/Transformer)在融合週期性特徵時,要麼在前期混淆特徵,要麼架構過於複雜、參數過多且計算成本高昂。

**核心創新:** 將人群流量預測轉化為週期性殘差學習問題。模型不直接預測高度動態的絕對流量,而是預測相對穩定的「未來流量與歷史同期觀測值之間的差值(殘差)」。

**三大貢獻:**
- **週期性殘差學習結構**:可輕易嵌入現有模型,顯著提升多步預測的準確度與魯棒性
- **時空增強網絡(SCE Encoder)**:一種輕量級的空間-通道增強編碼器,能高效捕捉全局空間關聯與通道動態特徵
- **優異的實驗表現**:在 TaxiBJ 和 BikeNYC 真實數據集上,相較於 SOTA 模型,MAE 降低了 $5.41\% \sim 17.63\%$,且參數減少了 $1.36 \sim 147.7$ 倍

---

## 🧮 核心公式詳細解析

本論文的公式主要分為兩個部分:殘差學習模組(解決週期性問題)與 SCE 編碼器(解決時空特徵提取問題)。

### 1. 時空特徵提取(時空模組)

每個時間段的模型輸入,都會先通過一個共享參數的時空網絡(ST Module)來轉換為高維特徵:

$$
h = f\left(\mathcal{P}^{t:t+T_{obs}}; \mathbf{W}_{st}\right)
$$

**(公式 1)**

**解析:**
- $\mathcal{P}^{t:t+T_{obs}}$:輸入的時間序列觀測段(如當前段、歷史同期段)
- $f(\cdot)$:代表時空網絡函數(本論文預設使用 SCE Encoder)
- $\mathbf{W}_{st}$:代表該網絡中所有可學習的參數
- $h \in \mathbb{R}^{H \times W \times C}$:輸出的高階時空特徵矩陣(包含長、寬、通道數)
- **特點**:所有時間段共享同一套參數,能大幅縮減記憶體與參數量的消耗

---

### 2. 差分函數(DIFF)

為了去除數據中的週期性季節特徵,模型計算「當前特徵」與「歷史週期特徵」之間的差值:

$$
\nabla_d \mathbf{H} = h_x - h_{px}
$$

**(公式 2)**

**解析:**
- $h_x$:當前時間段(Closeness)提取出的時空特徵
- $h_{px}$:歷史週期時間段(Periodic closeness,如上週同一時間)提取出的時空特徵
- $\nabla_d \mathbf{H}$:近期的週期性殘差(Closeness residual),它代表了當前流量相對於歷史規律的偏離程度
- **物理意義**:統計學中常透過「差分」來去除序列的非平穩性,這裡將其引入深度學習,讓模型訓練更容易、更穩定

---

### 3. 特徵融合函數(FUSE)

模型將近期的殘差特徵與未來的歷史週期特徵進行拼接,用以捕捉未來流量的偏移特徵:

$$
\tilde{\mathbf{H}} = \mathbf{W}_d \left(\nabla_d \mathbf{H} \parallel h_{py}\right)
$$

**(公式 3)**

**解析:**
- $\parallel$:矩陣拼接(Concatenation)操作
- $h_{py}$:未來預測目標的歷史同期觀測特徵(如預計預測明天下午,此處即為上週明天下午的特徵)
- $\mathbf{W}_d$:標準線性層的權重參數
- $\tilde{\mathbf{H}}$:預測殘差的隱藏狀態(Prediction residual features)。模型藉此學習未來真實流量與歷史同期的偏差趨勢

---

### 4. 損失函數與絕對流量還原

模型在訓練時,優化目標是預測差值與真實差值之間的平均絕對誤差(L1 Loss):

$$
\mathcal{L}(\theta) = \sum_{\tau=1}^{T_{pred}} \left\lVert \hat{\Delta \mathbf{Y}}_\tau - \Delta \mathbf{Y}_\tau \right\rVert_1
$$

**(公式 4)**

**解析:**
- $\hat{\Delta \mathbf{Y}}_\tau$:模型預測出的未來第 $\tau$ 個時段的流量偏差值
- $\Delta \mathbf{Y}_\tau$:未來真實的流量偏差值(真實值減去歷史同期值)

當模型輸出預測的偏差值 $\hat{\Delta \mathbf{Y}}$ 後,需透過以下公式將其還原為絕對人群流量 $\hat{\mathbf{Y}}$:

$$
\hat{\mathbf{Y}} = \sum_{i=1}^P \left(\hat{\Delta \mathbf{Y}} + \mathbf{Y}_p\right) / P
$$

**(公式 5)**

**解析:** 將預測的偏差加上歷史同期的基礎觀測值 $\mathbf{Y}_p$,並對選取的多個週期($P$ 個週期)求平均,以保證預測的穩定性與抗噪能力。

---

### 5. 空間-通道增強區塊(SCE Block)內部公式

為了更高效抓取時空關聯,論文設計了 SCE Block,包含三個模組:

**A. 標準 CNN 模組(局部特徵抓取)**

$$
h^{\circ(m)} = \mathbf{W}_f^{(m)2} \star \left( \delta \left( \mathbf{W}_f^{(m)1} \star h^{(m)} + b_f^{(m)1} \right) \right) + b_f^{(m)2}
$$

**(公式 6)**

解析:利用兩層卷積層($\star$ 為卷積操作)和 ReLU 激活函數($\delta$),提取局部的時空特徵,輸出為 $h^{\circ}$。

**B. 空間增強模組(SEM,全局空間抓取)**

$$
\hat{h}_s = \sigma(g(\mathbf{S}', \mathbf{W}_s)) = \sigma\left(\delta(\mathbf{S}' \mathbf{W}_{s1})\mathbf{W}_{s2}\right)
$$

**(公式 7)**

解析:先透過自適應最大池化(AMP)捕捉城市中最顯著的全局特徵 $\mathbf{S}'$。再通過兩層全連接層($\mathbf{W}_{s1}, \mathbf{W}_{s2}$)組成的門控機制進行維度壓縮與恢復,最後用 Sigmoid($\sigma$)激活,動態篩選出最重要的全局空間特徵。

**C. 通道增強模組(CEM,通道動態抓取)**

首先透過全局平均池化(GAP)將空間特徵壓縮為通道描述符 $c$:

$$
c = \frac{1}{H' \times W'} \sum_{h=1}^{H'} \sum_{w=1}^{W'} \tilde{h}_s(h, w)
$$

**(公式 8)**

再計算通道權重矩陣 $\tilde{h}$,用以重新校準各時空通道間的依賴關係:

$$
\tilde{h} = \sigma(g(c, \mathbf{W}_c)) = \sigma\left(\mathbf{W}_{c2} \, \delta(\mathbf{W}_{c1}c)\right)
$$

**(公式 9)**

**D. 最終特徵縮放融合**

$$
h^{(m+1)} = \tilde{h}^{(m)} \cdot h^{\circ(m)}
$$

**(公式 10)**

解析:將標準卷積特徵 $h^{\circ}$ 乘以通道注意力權重 $\tilde{h}$。多個區塊堆疊後,前層網路負責簡單的局部時空交互,後層網路則負責複雜的全局時空關聯。

---

## 📊 實驗結果與結論

1. **顯著超越基準模型**:在與 DeepST、ST-ResNet、ConvLSTM、Graph WaveNet 等 7 個基線模型的對比中,PRNet 在 MAE 和 RMSE 指標上均取得了最佳預測精確度
2. **極高的架構通用性(Generality)**:論文將該週期性殘差結構嵌入到其他老舊時空模型(如 DeepST+, ST-ResNet+, DeepLGR+)中,在參數減少的同時,使原模型的 MAE 誤差大幅降低了 $5.13\% \sim 14.77\%$
3. **小數據集下表現優異**:在僅給予 $10\%$ 的極少訓練數據預算下,PRNet 依然能保持高準確率(歸功於強大的歷史統計殘差先驗知識指導)
4. **超參數建議**:實驗證實「週週期(Weekly Scale)」的預測效果顯著優於「日週期(Daily Scale)」,且最佳的歷史參考週期數設為 $P=3$
