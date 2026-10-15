# Ôn vấn đáp và sửa model trực tiếp — CompactMix-L trên CIFAR-10/CIFAR-100

Tài liệu áp dụng các tình huống được nhắc trong đoạn trao đổi bạn cung cấp vào **đúng kiến trúc CompactMix-L của bài làm**. Mục tiêu là tự giải thích mạng, tính được shape/tham số và tự sửa các lớp liên quan để model tiếp tục forward, backward và train được.

Đối chiếu mã nguồn tại thời điểm viết: [compactmix_manual_train.py](compactmix_manual_train.py), notebook [CompactMix_CIFAR10_CIFAR100_BAO_CAO.ipynb](CompactMix_CIFAR10_CIFAR100_BAO_CAO.ipynb) và yêu cầu [Final exam.docx](../Final%20exam.docx). Các ví dụ sửa mạng bên dưới được thử trên model tạm trong bộ nhớ, **không thay kiến trúc đang lưu của bạn**.

## 1. Những việc cần chuẩn bị theo đoạn trao đổi

| Vấn đề được nhắc đến | Áp dụng vào bài CompactMix của bạn | Cần thể hiện khi được hỏi |
|---|---|---|
| Có bảng mô tả model | Mở bảng input/output shape, Params, MACs và cộng dồn toàn mạng | Đọc một dòng, thay số vào công thức và đối chiếu với code |
| Thầy thay stride của một khối | Theo dõi H,W ở hai nhánh, concat, residual và các stage sau | Tính shape trước khi sửa; tự sửa nhánh skip nếu cần |
| Thêm/bớt block | Thay cả khai báo trong `__init__` và đường đi trong `forward` | Giải thích shape có đổi không, P/FLOPs tăng giảm bao nhiêu |
| Đổi số kênh, tăng tham số vào khoảng yêu cầu | Sửa Conv, BN, chunk, groups, SE, project và lớp kế tiếp | Tìm đủ phụ thuộc; kiểm tra lại cả ≤6M và <1G |
| Giải thích kỹ thuật trong mạng | BN, SiLU, depthwise/grouped Conv, residual, SE, DropPath, GAP | Nêu tác dụng, lý do lựa chọn và đánh đổi |
| Hiểu code train và số liệu | SGD/Adam, LR, loss, augmentation, EMA, checkpoint, validation/test | Phân biệt cấu hình đã chạy, kế hoạch và kết quả đo thật |
| Sửa trực tiếp rồi chạy được | Kiểm tra forward → backward → budget → lượt train ngắn | Chứng minh code sau sửa hoạt động, không chỉ sửa hình minh họa |

**Phân biệt nguồn:** các tình huống sửa trực tiếp trên đây là nội dung ôn tập rút ra từ trao đổi trong ảnh. Thông tin như “chỉ có một lần thử” chưa được xác nhận là quy định chính thức của giảng viên; nên chuẩn bị kỹ nhưng không coi đây là điều khoản đã kiểm chứng.

Theo `Final exam.docx`, model L phải có **learnable parameters ≤6.000.000**, **FLOPs <1.000.000.000**, thực nghiệm cả CIFAR-10/CIFAR-100, dùng **SGD và Adam với các initial LR khác nhau**, ghi chi tiết learning settings và trình bày trong một PDF. Đề yêu cầu kiến trúc không trùng nhóm khác/không hoàn toàn giống mạng trước, với ngoại lệ model giữa kỳ của chính nhóm theo đề. Bài làm chưa có bằng chứng xác nhận tính độc nhất này.

Thang điểm ghi trong đề: vấn đáp 30%, CIFAR-10 30%, CIFAR-100 30%, PDF 10%; không qua vấn đáp thì trượt môn. Đề **không ấn định** 600 epoch, split 45k/5k, retrain 50k, hay accuracy bắt buộc 98%/90%.

## 2. Cách giới thiệu mạng trong khoảng một phút

> Em dùng CompactMix-L, một mạng CNN nhận ảnh RGB 32×32. Stem là Conv 3×3 từ 3 lên 40 kênh. Mạng có ba stage với số kênh 40, 112, 256 và số block lần lượt 2, 3, 8; hai transition giảm kích thước không gian từ 32 xuống 16 rồi 8. Trong mỗi block, Conv 1×1 mở rộng kênh, chia đôi để xử lý bằng depthwise 3×3 và 5×5, nối lại rồi dùng grouped Conv. SE tạo trọng số theo kênh; projection đưa số kênh về mức ban đầu để cộng residual qua DropPath. Cuối mạng dùng global average pooling, flatten và Linear ra 10 hoặc 100 logits. Với một ảnh 32×32, bản CIFAR-10 có 5.248.172 tham số và khoảng 0,8424 GFLOPs, đạt ngân sách của đề. Hai kết quả test đã lưu là 97,18% và 82,98% cho best của lượt train 45k, không phải kết quả của model sửa trực tiếp.

Lời giới thiệu cần đi cùng khả năng chỉ đúng từng lớp và giải thích công thức ở các phần sau.

## 3. Luồng tổng thể và bảng phải đọc được

```text
Input                     [N,  3,32,32]
  ↓ stem: Conv3×3 3→40    [N, 40,32,32]  (stem không có BN)
  ↓ Block01, Block02      [N, 40,32,32]  Stage 1: 2 block
  ↓ t1_bn → SiLU → AvgPool2×2,s2 → Conv1×1 40→112
                          [N,112,16,16]
  ↓ Block03..Block05      [N,112,16,16]  Stage 2: 3 block
  ↓ t2_bn → SiLU → AvgPool2×2,s2 → Conv1×1 112→256
                          [N,256, 8, 8]
  ↓ Block06..Block13      [N,256, 8, 8]  Stage 3: 8 block
  ↓ head_bn → SiLU        [N,256, 8, 8]
  ↓ AdaptiveAvgPool2d(1)  [N,256, 1, 1]
  ↓ flatten(1)            [N,256]
  ↓ Dropout(p=0)          [N,256]
  ↓ Linear(256,K)         [N,K]         K=10 hoặc 100; logits
```

`N` là số ảnh trong batch, `C` là số kênh, `H` là chiều cao, `W` là chiều rộng. Train đã lưu dùng batch 128; profile tính cho **N=1**; kiểm tra backward của file Python dùng batch 2. Không lấy chi phí batch 128 so trực tiếp với FLOPs/ảnh trong báo cáo.

| Thành phần | Input → output, bỏ N | Số lần | Params mỗi thành phần | MACs mỗi thành phần |
|---|---|---:|---:|---:|
| Stem | 3×32×32 → 40×32×32 | 1 | 1.080 | 1.105.920 |
| Block Stage 1 | 40×32×32 → 40×32×32 | 2 | 16.888 | 15.320.320 |
| Transition 1 | 40×32×32 → 112×16×16 | 1 | 4.560 | 1.146.880 |
| Block Stage 2 | 112×16×16 → 112×16×16 | 3 | 118.510 | 28.276.864 |
| Transition 2 | 112×16×16 → 256×8×8 | 1 | 28.896 | 1.835.008 |
| Block Stage 3 | 256×8×8 → 256×8×8 | 8 | 602.656 | 36.241.408 |
| Head BN/SiLU/GAP/flatten/dropout | 256×8×8 → 256 | 1 | 512 | 0 Conv/Linear MACs |
| Linear CIFAR-10 | 256 → 10 | 1 | 2.570 | 2.560 |
| Linear CIFAR-100 | 256 → 100 | 1 | 25.700 | 25.600 |

| Bản gốc | Learnable params | MACs/ảnh | FLOPs/ảnh | Đạt đề? |
|---|---:|---:|---:|---|
| CIFAR-10 | 5.248.172 | 409.492.864 | 842.406.944 | Có |
| CIFAR-100 | 5.271.302 | 409.515.904 | 842.453.114 | Có |

**Cách đọc dòng Conv đầu:** `Conv3×3: 3→40, s=1,p=1,g=1,b=0,d=1` nằm ở `CompactMixManual.__init__`, thuộc `self.stem`; trong forward là `x = self.stem(x)`. Đây là stem trước Block01, không phải `Block01.local`. `b=0` nghĩa là không có bias, `d=1` là dilation 1. P=40×3×3×3=1.080, output 40×32×32.

Bảng đầy đủ từng block và cộng dồn: [BANG_CONG_DON_THAM_SO_MACS.md](BANG_CONG_DON_THAM_SO_MACS.md); hình [CIFAR-10](figures/BANG_CONG_DON_CIFAR10.png) và [CIFAR-100](figures/BANG_CONG_DON_CIFAR100.png).

## 4. Bóc tách một block: Block01

Các block cùng cấu trúc nhưng có số kênh khác nhau. Không gọi Block01 có 40 kênh nghĩa là mọi lớp bên trong đều 40 kênh.

| Thứ tự/lớp | Thông số chính | Output |
|---|---|---|
| Input/skip | Lưu `x` để cộng cuối block | [N,40,32,32] |
| `bn_pre → act_pre` | BN(40), eps=1e-5, momentum=0,1 → SiLU | [N,40,32,32] |
| `expand` | Conv 1×1 40→80, s1,p0,g1, bias=False | [N,80,32,32] |
| `bn_expand → act_expand` | BN(80) → SiLU | [N,80,32,32] |
| `a,b = y.chunk(2,dim=1)` | Chia 80 kênh thành 40+40 | Mỗi nhánh [N,40,32,32] |
| `local(a)` | DW Conv 3×3 40→40, s1,p1,g40, bias=False | [N,40,32,32] |
| `context(b)` | DW Conv 5×5 40→40, s1,p2,g40, bias=False | [N,40,32,32] |
| `concat_checked` | Nối theo dim=1: 40+40=80 | [N,80,32,32] |
| `bn_mix → act_mix` | BN(80) → SiLU | [N,80,32,32] |
| `grouped` | Conv 3×3 80→80, s1,p1,g8, bias=False | [N,80,32,32] |
| `bn_gate → act_gate` | BN(80) → SiLU; lưu `feature` | [N,80,32,32] |
| `se_pool` | AdaptiveAvgPool2d(1) | [N,80,1,1] |
| `se_reduce → se_act` | Conv 1×1 80→8, bias=True → SiLU | [N,8,1,1] |
| `se_expand → se_sigmoid` | Conv 1×1 8→80, bias=True → Sigmoid | [N,80,1,1] |
| `multiply_checked` | feature × gate, broadcast theo H,W | [N,80,32,32] |
| `project` | Conv 1×1 80→40, s1,p0,g1, bias=False | [N,40,32,32] |
| `drop` | DropPath; Block01 có p=0 | [N,40,32,32] |
| `residual_checked(x,y)` | Cộng hai tensor cùng shape | [N,40,32,32] |

Ở Stage 2: C=112, expand E=224, SE bottleneck D=14. Ở Stage 3: C=256, E=512, D=32. Trong bản gốc, E=2C; D được khai báo bằng số trong từng class.

## 5. Công thức và điều kiện kết nối cần nhớ

### 5.1. Conv 2d

Với từng chiều riêng biệt:

```text
Hout = floor((Hin + 2*pH - dH*(kH-1) - 1)/sH + 1)
Wout = floor((Win + 2*pW - dW*(kW-1) - 1)/sW + 1)
Cout = out_channels
Pweight = Cout * (Cin/groups) * kH * kW
Ptotal = Pweight + Cout nếu bias=True; ngược lại chỉ Pweight
MACs/ảnh = Hout * Wout * Cout * (Cin/groups) * kH * kW
```

`kH,kW` là chiều cao/rộng của kernel. `(Cin/groups)` là phép chia số kênh vào cho số nhóm, không phải ký hiệu một biến mới. Cần `Cin % groups == 0` và `Cout % groups == 0`. Floor là lấy số nguyên lớn nhất không vượt quá giá trị: floor(1,5)=1; floor(-1,5)=-2.

Depthwise trong bài: `groups=Cin=Cout`, nên Pweight=Cin×kH×kW. Grouped Conv 80→80,g8 có mỗi nhóm 10 kênh vào và 10 kênh ra. Conv 1×1,g1 trộn kênh, s1,p0 giữ H,W. Với kernel lẻ, d1,s1, đặt p=(k−1)/2 để giữ H,W.

### 5.2. Các phép còn lại

| Phép | Quy tắc shape/tham số | Lỗi cần tránh |
|---|---|---|
| `BatchNorm2d(C)` | Giữ [N,C,H,W]; num_features=C; mặc định có 2C tham số học gamma/beta | Đổi Conv ra 96 nhưng giữ BN 80 |
| SiLU, Sigmoid, Dropout, DropPath | Giữ shape; không có tham số học | Tưởng DropPath làm tensor còn ít kênh hơn |
| `chunk(2,dim=1)` | Với C chẵn: mỗi nhánh [N,C/2,H,W] | C lẻ có thể chia không đều; không mặc định hai nhánh bằng nhau |
| `cat((a,b),dim=1)` | Cộng số kênh; N,H,W phải giống nhau | Một nhánh stride 2, nhánh kia stride 1 |
| Residual ADD của bài | Hai tensor phải cùng toàn bộ [N,C,H,W] | Dựa vào broadcasting để che lỗi residual |
| SE multiply của bài | feature [N,E,H,W], gate đúng [N,E,1,1]; broadcast H,W | SE trả 8 kênh rồi nhân feature80 kênh |
| `AvgPool2d(2,2)` | Mặc định floor: Hout=floor(H/2), Wout=floor(W/2); giữ C | Cho rằng pool giảm số kênh |
| `AdaptiveAvgPool2d((a,b))` | [N,C,H,W] → [N,C,a,b], không giảm C | Nhầm SE giảm80→8 là do pooling; việc đó thuộc `se_reduce` |
| `flatten(1)` | [N,C,H,W] → [N,C×H×W] | Pool2×2 nhưng Linear vẫn nhận C |
| `Linear(Fin,Fout)` | [N,Fin] → [N,Fout]; P=Fin×Fout+Fout nếu có bias | Đổi kênh cuối nhưng quên in_features |

PyTorch nói chung cho ADD/multiply dùng broadcasting nếu các chiều tương ứng bằng nhau hoặc một chiều là 1, xét từ phải sang trái. **Code bài chủ động kiểm tra chặt hơn**: residual phải trùng shape, còn SE gate phải đúng N,E,1,1. Điều này giúp phát hiện chỗ nối sai.

### 5.3. Params, MACs, FLOPs khác nhau thế nào?

- Params là số giá trị học được, không phụ thuộc batch/H/W đối với Conv thông thường. BN running mean/variance là buffer, không cộng vào learnable params.
- MAC là một phép nhân tích lũy. Bảng hiện tại chỉ tính MACs Conv/Linear.
- Quy ước báo cáo: phần Conv/Linear tính 1 MAC = 2 FLOPs, cộng thêm bias và chi phí BN/activation/pooling/SE multiply/residual ADD. Vì vậy **tổng FLOPs lớn hơn 2×MACs**, không lấy 409,49 triệu MAC làm 409,49 triệu FLOPs.
- Đây là ước lượng inference của profiler với input một ảnh; không phải thời gian chạy thực, không phải chi phí backward. Thời gian còn phụ thuộc GPU, batch, precision và cách đo.
- Với block gốc C,E,D và H,W: `P=2CE+34C+9E²/8+2ED+(D+E)+2(C+3E)`; `MAC=HW×(2CE+34C+9E²/8)+2ED`. Công thức này chỉ áp dụng khi kernel/groups/stride giống block gốc.

## 6. Quy trình sửa trực tiếp khi thầy đưa yêu cầu

1. **Nhắc lại thay đổi cụ thể:** lớp/block nào, đổi kênh/kernel/stride hay thêm block, mục tiêu tham số, có cần giữ output shape hay không.
2. **Viết shape trước và sau trên giấy:** đặc biệt tại chunk → hai nhánh → concat → SE → project → ADD.
3. **Liệt kê lớp liên quan rồi sửa bằng tay:** Conv ra E thì BN theo E, chunk theo E/2, groups hợp lệ, SE trả E, project/skip khớp, input lớp sau nhận đúng kênh.
4. **Lưu và tạo model mới:** chạy lại class/lớp liên quan và constructor; model đã tạo không tự cập nhật khi bạn sửa class.
5. **Chạy forward và backward**, sau đó xem Params/MACs/FLOPs. Sửa đúng lỗi đầu tiên xuất hiện rồi kiểm tra lại; đừng chỉ nhìn lớp cuối báo lỗi.
6. **Dùng tên run mới** nếu đổi source/recipe/model. Thử train ngắn để chứng minh pipeline hoạt động. Không nạp checkpoint mạng cũ vào mạng đã sửa.
7. **Cập nhật bảng/sơ đồ tương ứng** và nói rõ đây là biến thể mới, chưa có accuracy test mới.

Mẹo tìm code: tìm `class Block01`, `class CompactMixManual`, `def forward`, `self.t1_conv`, `self.classifier`, `TRAIN_SETTINGS`. File `.py` và các cell notebook là **hai bản mã riêng**; sửa file Python không tự sửa notebook.

## 7. Tình huống A — đổi kernel context của Block01 từ 5×5 thành 7×7

Sửa trong `Block01.__init__`:

```python
self.context = nn.Conv2d(
    in_channels=40, out_channels=40, kernel_size=7,
    stride=1, padding=3, dilation=1, groups=40, bias=False,
)
```

Từ kernel 5,p2 sang kernel 7,p3, H,W vẫn32×32. Không phải sửa BN, concat, SE hay project vì số kênh và kích thước output giữ nguyên. Nếu chỉ đổi kernel 7 mà giữ padding 2, output nhánh context còn30×30, không concat được với local32×32.

Tăng P=40×(7²−5²)=960; tăng MAC=32×32×960=983.040; tăng FLOPs=1.966.080. Không kết luận kernel lớn hơn chắc chắn tăng accuracy: receptive field lớn hơn nhưng tốn tính toán hơn và cần thực nghiệm.

## 8. Tình huống B — đổi stride trong Block01

### 8.1. Chỉ đổi `context.stride=2`

Local trả [N,40,32,32]; context5×5,p2,s2 trả [N,40,16,16]. Kết quả: lỗi `Block01.concat`. Phải làm hai nhánh có cùng H,W theo thiết kế mong muốn.

### 8.2. Đổi `expand.stride=2`

Expand giảm32→16 trước khi chunk nên cả local/context và concat đều hợp lệ. Tuy nhiên project trả [N,40,16,16], trong khi skip `x` còn [N,40,32,32]. Kết quả: lỗi `Block01.ADD`.

Nếu mục tiêu là **Block01 thực sự downsample**, một cách sửa rõ ràng với input32×32 là thêm skip pooling:

```python
# Trong Block01.__init__: thay expand, thêm skip_pool.
self.expand = nn.Conv2d(40, 80, kernel_size=1, stride=2, padding=0, bias=False)
self.skip_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Trong Block01.forward: giữ các phép trên nhánh chính; thay dòng return cuối.
return residual_checked(self.skip_pool(x), y, "Block01.ADD")
```

Hai phía ADD đều [N,40,16,16]. Không đưa tensor đã pool vào đầu nhánh chính, vì expand đã có stride 2. Các stage sau lần lượt: Stage 1 16×16 → transition1 8×8 → Stage 2 8×8 → transition2 4×4 → Stage 3 4×4 → GAP1×1 → Linear; vẫn chạy được.

P không đổi vì stride/pooling không thêm weight; FLOPs giảm mạnh. Với kích thước lẻ, Conv 1×1,s2 và AvgPool2,s2 có thể ra kích thước khác nhau: phải tính lại, không áp dụng mù quáng ví dụ32×32 này.

Một phương án khác là **skip Conv 1×1,stride 2** nếu cần đổi cả số kênh, nhưng sẽ thêm tham số/tính toán. Nếu thầy yêu cầu khối giữ32×32 thì cần thiết kế lại việc downsample phù hợp, không chỉ thêm skip pooling.

Bản gốc Block01 có stride 1. Nếu đề miệng nói “đổi về stride 1”, cần chỉ rõ lớp hiện đang stride 2; không nói đã sửa khi giá trị vốn là1.

## 9. Tình huống C — tăng Stage 1 từ 40 lên 48 kênh

Mục tiêu: giữ Stage 1 ở32×32, vẫn dùng hai nhánh bằng nhau và expand gấp đôi. Stage 2/3 giữ112/256 kênh. Cần sửa **cả Block01 và Block02**, cùng stem và đầu transition1.

### 9.1. Sửa phần mạng tổng

```python
# Trong CompactMixManual.__init__:
self.stem = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False)
self.t1_bn = nn.BatchNorm2d(48, eps=1e-5, momentum=0.1)
self.t1_conv = nn.Conv2d(48, 112, kernel_size=1, stride=1, padding=0, bias=False)
```

### 9.2. Sửa từng khai báo trong cả Block01 và Block02

| Thuộc tính | Bản gốc | Sau sửa |
|---|---|---|
| `bn_pre.num_features` | 40 | 48 |
| `expand` | Conv 1×1 40→80 | Conv 1×1 48→96 |
| `bn_expand.num_features` | 80 | 96 |
| `local` | DW 3×3 40→40,g40,p1 | DW 3×3 48→48,g48,p1 |
| `context` | DW 5×5 40→40,g40,p2 | DW 5×5 48→48,g48,p2 |
| `bn_mix.num_features` | 80 | 96 |
| `grouped` | Conv 3×3 80→80,g8,p1 | Conv 3×3 96→96,g8,p1 |
| `bn_gate.num_features` | 80 | 96 |
| `se_reduce` | Conv 1×1 80→8,bias=True | Conv 1×1 96→8,bias=True |
| `se_expand` | Conv 1×1 8→80,bias=True | Conv 1×1 8→96,bias=True |
| `project` | Conv 1×1 80→40 | Conv 1×1 96→48 |

Giữ stride 1; Conv expand/project padding 0; các Conv ngoài SE vẫn bias=False. Các SiLU, Sigmoid, pooling SE, chunk và forward không cần đổi. Giữ SE bottleneck 8 là một lựa chọn hợp lệ của ví dụ này, không tự động đổi thành6 hoặc12.

Sau expand96, chunk tạo48+48; grouped 96 chia hết8; SE trả96; project trả48 để cộng skip48. Transition1 nhận48 rồi trả112 nên không phải đổi Block03 trở đi. Không thay tất cả số40/80 trong toàn file bằng thao tác tìm-thay thiếu kiểm soát.

P toàn mạng tăng14.408; FLOPs tăng27.847.872 so với gốc. Việc chỉ sửa `stem.out_channels=48` sẽ lỗi ngay ở `block01.bn_pre`, vì BN vẫn chờ40.

**Nếu chỉ muốn tăng số kênh ẩn bên trong Block01**, có thể giữ input/output40 và đổi expand80→96, các nhánh thành48, BN/grouped/SE theo96, project96→40. Khi đó không cần đổi stem/Block02/transition1. Đây là một biến thể khác với đổi cả Stage 1; phải đo riêng, không dùng số đo của ví dụ48 kênh toàn stage.

## 10. Tình huống D — thêm một block Stage 3 để tăng khoảng 0,6 triệu tham số

1. Sao chép đầy đủ class `Block13` thành class **`Block14(ManualBlock)`**; giữ nguyên mọi Conv/BN/SE và forward256→256. Đổi các nhãn lỗi `Block13.*` thành `Block14.*`; có thể giữ DropPath 0,1 như block cuối gốc.
2. Trong `CompactMixManual.__init__`, sau `self.block13 = Block13()` thêm `self.block14 = Block14()`.
3. Trong `CompactMixManual.forward`, sau `x = self.block13(x)` và trước `head_bn`, thêm `x = self.block14(x)`.
4. Tạo lại model và kiểm tra. Input/output của block mới đều [N,256,8,8], không cần sửa head/Linear.

Tăng **602.656 tham số**, **36.241.408 MACs**, **73.714.912 FLOPs**. Cả hai dataset vẫn đạt ngân sách; xem bảng kiểm chứng ở mục13.

Không dùng `self.block14 = self.block13` nếu mục tiêu là thêm tham số độc lập: đó là cùng một module dùng chung weight. Gọi `self.block13(x)` hai lần cũng tăng tính toán nhưng không tạo thêm bộ tham số.

Nếu thêm **hai** block Stage 3 độc lập, CIFAR-10 có6.453.484 và CIFAR-100 có6.476.614 tham số: **vượt6M**, dù FLOPs vẫn dưới1G. Vì vậy phải kiểm tra đồng thời hai giới hạn.

Để chọn nơi thêm block, số tăng mỗi block gốc là:

| Stage | ΔParams | ΔFLOPs |
|---|---:|---:|
| Stage 1, 40 kênh/32×32 | 16.888 | 33.713.176 |
| Stage 2, 112 kênh/16×16 | 118.510 | 58.705.570 |
| Stage 3, 256 kênh/8×8 | 602.656 | 73.714.912 |

Nếu cần khoảng Pmin…Pmax, trước hết tính `Pmin−Pgốc ≤ ΔP ≤ Pmax−Pgốc`, rồi kiểm tra ngân sách đề trên cả hai dataset. Thêm nhiều block đầu mạng có thể chạm FLOPs trước khi chạm Params. Không nói “dưới6M là đủ”.

## 11. Tình huống E — bỏ Block13

Xóa hoặc bỏ đồng thời:

```python
# Trong __init__:
self.block13 = Block13()

# Trong forward:
x = self.block13(x)
```

Đây là hai dòng cần bỏ, không phải đoạn code để thêm. Block12 đã trả [N,256,8,8], nên có thể nối thẳng vào head. Giảm602.656 tham số và73.714.912 FLOPs.

Nếu chỉ bỏ dòng trong forward nhưng vẫn giữ module trong `__init__`, `sum(p.numel() for p in model.parameters())` **vẫn đếm các weight không được dùng**. Không tuyên bố số tham số đã giảm khi chỉ ngừng gọi block. Class Block13 còn tồn tại nhưng không được khởi tạo thì không tạo tham số trong model.

## 12. Tình huống F — đổi head pooling1×1 thành2×2

```python
self.head_pool = nn.AdaptiveAvgPool2d(output_size=2)
self.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
```

Luồng mới: [N,256,8,8] → [N,256,2,2] → flatten → [N,1024]. Vì256×2×2=1024 nên phải sửa Linear; pooling vẫn giữ256 kênh. Nếu chỉ đổi pool mà giữ Linear 256 sẽ lỗi nhân ma trận.

So với Linear gốc, tăng768×K tham số:7.680 với K10 hoặc76.800 với K100. Không thay đổi `head_bn=256` vì BN vẫn nhận256 kênh.

## 13. Số đo kiểm chứng các ví dụ sửa mạng

Các biến thể trong bảng là **những thay đổi độc lập, mỗi lần bắt đầu từ bản gốc**. Không cộng dồn tất cả thay đổi rồi dùng lại bảng này. Đã kiểm tra CPU forward và backward với cả K10/K100 bằng đúng Microsoft Store Python 3.13; profiler chạy input [1,3,32,32]. **Chưa train để đo accuracy của các biến thể.**

| Biến thể | Params C10 | FLOPs C10 | Params C100 | FLOPs C100 | Forward/backward và budget |
|---|---:|---:|---:|---:|---|
| Gốc | 5.248.172 | 842.406.944 | 5.271.302 | 842.453.114 | Đạt |
| Block01 context7×7,p3 | 5.249.132 | 844.373.024 | 5.272.262 | 844.419.194 | Đạt |
| Stage 1 rộng48, SE giữD8 | 5.262.580 | 870.254.816 | 5.285.710 | 870.300.986 | Đạt |
| Thêm1 block Stage 3 độc lập | 5.850.828 | 916.121.856 | 5.873.958 | 916.168.026 | Đạt |
| Bỏ Block13 cả khai báo/lời gọi | 4.645.516 | 768.692.032 | 4.668.646 | 768.738.202 | Đạt |
| Block01 expand s2 + skip AvgPool2 | 5.248.172 | 213.072.160 | 5.271.302 | 213.118.330 | Đạt |
| Head pool 2×2 + Linear 1024 | 5.255.852 | 842.423.072 | 5.348.102 | 842.607.482 | Đạt |
| Thêm2 block Stage 3 độc lập | 6.453.484 | 989.836.768 | 6.476.614 | 989.882.938 | **Bị từ chối vì vượt Params** |

Các số FLOPs tuân theo convention profiler hiện có, không phải số chu kỳ phần cứng. Hai lỗi chưa sửa ở mục8 cũng đã được tái hiện đúng tại `Block01.concat` và `Block01.ADD`. Kiểm tra kỹ thuật đạt không chứng minh biến thể có accuracy tốt hơn hay đã đáp ứng đầy đủ thực nghiệm của đề.

## 14. Chạy kiểm tra và minh họa train bằng một file Python

Mở PowerShell tại `D:\DeepLearning\Finnal`. Dùng đúng Python hiện tại; không tạo môi trường mới.

### 14.1. Forward/backward, shape từng lớp và ngân sách

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 report/compactmix_manual_train.py --check --dataset both
```

Lệnh không train dữ liệu thật, không cần mở test, không tự sửa shape. Thành công phải có `FORWARD + BACKWARD OK` cho cả hai dataset. Chế độ kiểm tra in nhiều lớp để chẩn đoán; quy định “một dòng sau mỗi epoch” áp dụng cho chế độ train thông thường.

### 14.2. Lượt thử kỹ thuật2 epoch trên FakeData

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 -u report/compactmix_manual_train.py --smoke --dataset both --epochs 2 --run-name van_dap_sua_01
```

Tên `van_dap_sua_01` phải chưa dùng cho code/config khác; đổi hậu tố khi tạo biến thể khác. FakeData chỉ kiểm tra train/validation/checkpoint, **accuracy của nó không có ý nghĩa chất lượng CIFAR**. Nếu run đã đủ2 epoch, chạy lại sẽ không tạo thêm epoch.

### 14.3. Minh họa một epoch CIFAR thật mà giữ nguyên tổng lịch

Chỉ tự chạy khi cần trình diễn và có đủ thời gian:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 -u report/compactmix_manual_train.py --dataset cifar10 --selection-only --run-name van_dap_cifar10_01 --stop-after-epochs 1
```

Lệnh dùng dữ liệu có sẵn trong `datasets/`, train/validation một epoch mới rồi dừng ở ranh giới epoch, không chuyển sang retrain/test. Tổng schedule vẫn là cấu hình600 epoch; đây là một minh họa khác với chọn toàn bộ schedule chỉ2 epoch. Lặp đúng lệnh, code và config sẽ tiếp tục thêm một epoch từ `last.pt`.

Không đổi `--epochs` để điều khiển số epoch mới khi resume vì tổng schedule thuộc config. Đổi source/model/optimizer/LR thì dùng run-name mới; không xóa best/last cũ để ép resume. Mỗi run chỉ giữ `best.pt` và `last.pt`.

## 15. Khi trình bày bằng notebook: chạy lại cell nào?

1. Chọn kernel Microsoft Store Python 3.13 đang dùng. Giữ các cờ train matrix/final/test lại ở trạng thái tắt nếu chỉ trình bày ngắn.
2. Với bản gốc, chạy các cell từ đầu đến định nghĩa/profile/bảng/sơ đồ; đọc phần archived evidence để trình bày số liệu đã lưu. Notebook mặc định có chế độ demo dữ liệu con; số đo demo phải gọi đúng là demo.
3. Khi sửa Block01: chạy lại cell **định nghĩa Block01**; nếu sửa Block02/transition/head thì chạy lại các cell định nghĩa tương ứng và **cell định nghĩa toàn mạng**.
4. Chạy lại **cell profile** để tạo model mới rồi kiểm tra forward/backward/budget. Việc chạy lại class không làm các model/optimizer đã tồn tại thay đổi. Nếu dùng biến model riêng thì phải khởi tạo lại model và optimizer của biến đó.
5. Đổi **RUN_NAME** cho biến thể mới trước khi train; giữ tổng epochs/source/recipe nếu mục tiêu là resume cùng run.
6. Nếu thêm/bớt block, chỉnh cả các danh sách tổng hợp và sơ đồ. **Ngay trong cell `profile-results`**, danh sách `prefix` đang dùng `range(6,14)`; thêm Block14 thì đổi thành `range(6,15)`, bỏ Block13 thì đổi thành `range(6,13)`, để phép kiểm tra tổng tham số không bỏ sót block mới. Hàm `cumulative_model_accounting` ở cell bảng cộng dồn cũng có danh sách `names` dùng `range(6,14)` cần sửa tương ứng. Cập nhật cả mô tả độ sâu2/3/8. **Forward/profile của model chạy được không có nghĩa bảng tổng hợp cũ tự nhận mọi block mới.**
7. Sau khi sửa kiến trúc, **bỏ qua các cell nạp/test checkpoint archived và xuất PDF gắn với evidence cũ**. Adapter weights và báo cáo lưu trữ chỉ dành cho kiến trúc gốc; không dùng `strict=False` để lờ tensor không khớp. Muốn báo cáo biến thể mới phải có evidence mới phù hợp.

Thứ tự trình bày thuận tiện: **giới thiệu bản gốc và kết quả đã lưu → làm yêu cầu sửa mạng → chạy profile/check → chạy một lượt demo mới**. Tránh `Run All` mù quáng sau sửa vì các cell cuối có thể phụ thuộc cấu trúc gốc. Muốn trở lại bản gốc, phục hồi các khai báo bạn đã sửa và chạy lại định nghĩa/model; không ghép bảng cũ với code mới.

Nếu tạo cell riêng cho Block14, đầu cell phải là `_remember_cell("block14")`, không giữ nhầm tên `block13` của cell sao chép. Trong `source_fingerprint`, cập nhật danh sách cell bắt buộc từ `range(1,14)` thành `range(1,15)`. Nếu bỏ hẳn cell Block13 khi giảm độ sâu thì danh sách bắt buộc phải bỏ nó; có thể giữ cell/class Block13 nhưng không khởi tạo trong model. Restart Kernel rồi chạy lại các định nghĩa theo thứ tự giúp loại source/model cũ còn trong bộ nhớ; khi trở về bản13 block cũng phải phục hồi các danh sách tương ứng.

## 16. Tác dụng, lý do lựa chọn và đánh đổi của từng kỹ thuật

Đây là giải thích cơ chế và động cơ thiết kế. Bài chưa có ablation đủ để gán số phần trăm accuracy tăng riêng cho từng thành phần.

| Kỹ thuật | Cơ chế và lý do dùng trong bài | Đánh đổi/điều cần nói đúng |
|---|---|---|
| Conv 3×3 stem, stride 1 | Lấy đặc trưng ban đầu, giữ32×32 vì ảnh CIFAR nhỏ | Tốn tính toán hơn giảm kích thước ngay đầu |
| Conv 1×1 expand/project | Trộn thông tin kênh, mở rộng khả năng biểu diễn rồi đưa về C để residual | Chi phí tăng theo tích Cin×Cout |
| Depthwise3×3 và5×5 | Mỗi kênh xử lý riêng, hai nhánh có phạm vi kernel khác nhau với ít weight | Không tự trộn thông tin giữa các kênh; cần các Conv khác |
| Grouped Conv,g8 | Trộn thông tin trong từng nhóm với chi phí thấp hơn Conv thường cùng kênh/kernel | Kết nối giữa nhóm bị hạn chế; Cin/Cout phải chia hết8 |
| BN | Chuẩn hóa từng kênh theo thống kê batch/spatial rồi affine gamma/beta, giúp quá trình tối ưu ổn định | Train/eval dùng thống kê khác nhau; batch quá nhỏ có thể làm thống kê kém |
| SiLU | `x*sigmoid(x)`, phi tuyến trơn, cho phép giá trị âm nhỏ | Không phải ReLU; không có weight học riêng, tốn phép tính hơn ReLU theo convention |
| Residual | `output=x+F(x)`, cung cấp đường truyền tín hiệu/gradient trực tiếp | Phải khớp shape; không bảo đảm cứ sâu hơn là tốt hơn |
| SE | GAP lấy mô tả kênh, bottleneck rồi Sigmoid sinh gate, nhân lại feature | Có thêm Conv/weight; không gọi đây là self-attention Transformer |
| DropPath | Khi train, ngẫu nhiên bỏ cả nhánh residual theo từng ảnh, chia cho xác suất giữ để bảo toàn kỳ vọng | Giữ shape; eval là identity; khác Dropout từng phần tử |
| AvgPool ở transition | Giảm H,W để tiết kiệm tính toán phía sau; Conv 1×1 đổi kênh | Mất chi tiết không gian, pool không tự tăng/giảm C |
| GAP ở head | Lấy trung bình không gian mỗi kênh, tạo256 features trước Linear | Mất vị trí chi tiết; đổi pool 2×2 phải sửa Linear |
| Dropout ở head | Khi p>0 có thể regularize các features | Bản này p=0 nên không thực sự bỏ phần tử |

**BN cần giải thích thêm:** `eps=1e-5` được cộng vào variance trước căn để ổn định phép chia. `momentum=0.1` của BN cập nhật running statistics theo `new=0.9*old+0.1*batch_stat`; không phải momentum 0,9 của optimizer. Khi `model.train()` dùng thống kê batch; khi `model.eval()` mặc định dùng running statistics. `torch.no_grad()` tắt ghi graph, không thay thế `eval()`.

## 17. Hiểu code train và trả lời về thực nghiệm

### 17.1. Những thiết lập đã dùng

| Nội dung | Lượt train gốc đã lưu |
|---|---|
| Dữ liệu mỗi tập | 45.000 train +5.000 validation, tách từ50.000 train chính thức; test10.000 riêng |
| Kiến trúc | 2/3/8 block, kênh40/112/256; Linear 10 hoặc100 |
| Epoch/batch | 600/128; đây là lựa chọn của bài, không phải số bắt buộc trong đề |
| Optimizer | SGD, initial LR 0,1, momentum 0,9, Nesterov=True |
| Weight decay | 0,0005 cho nhóm tham số ndim>1; nhóm còn lại0 |
| LR schedule | Warmup5 epoch rồi cosine |
| Augmentation | Crop/flip/AutoAugment, Mixup α0,2, CutMix α1,0, xác suất mixing0,75; mixing tắt ở đoạn cuối lịch |
| Loss | Soft-label cross entropy; label smoothing 0,05 |
| EMA/gradient clipping | EMA decay tối đa0,999; clip grad norm5 |
| Precision/seed | CUDA AMP, deterministic=True, model seed 42, split seed 2026 |
| Regularization trong model | DropPath tăng0→0,1 theo block; head dropout 0 |
| Checkpoint/log | Chỉ cập nhật best/last, metrics và dòng log sau epoch train+validation hoàn chỉnh, ghi thành công |

Workers của các run gốc là2; notebook dùng0 để phù hợp Dataset định nghĩa trong cell trên Windows. Cấu hình demo và dữ liệu con của notebook không đại diện cho 600 epoch đã train.

### 17.2. Các câu thường được hỏi

| Câu hỏi | Nội dung cần trả lời |
|---|---|
| Vì sao không thêm Softmax vào cuối forward? | Model trả logits. Cross entropy/log-softmax xử lý chuẩn hóa bên trong. Khi cần xác suất để trình bày mới lấy softmax; argmax logits và softmax giống nhau. |
| Loss và accuracy khác gì? | Loss dùng để tối ưu và phản ánh phân bố dự đoán; accuracy đếm dự đoán đúng. Loss giảm không bảo đảm mỗi lần accuracy đều tăng. |
| Mixup là gì? | Trộn hai ảnh theo λ, đồng thời trộn nhãn theo cùng trọng số; tạo soft targets. |
| CutMix khác Mixup ở đâu? | Thay một vùng ảnh bằng vùng của ảnh khác; trọng số nhãn theo diện tích thực tế sau cắt/giới hạn biên. |
| Label smoothing làm gì? | Làm mềm nhãn `target*(1−ε)+ε/K`, giảm ép xác suất vào đúng một lớp tuyệt đối. |
| Train accuracy có so trực tiếp với val được không? | Train dùng ảnh/nhãn trộn và mixed accuracy có trọng số; val/test dùng ảnh sạch, nhãn cứng. Không xem đây là cùng một phép đo. |
| SGD khác Adam? | SGD cập nhật theo gradient, có momentum trong bài; Adam dùng trung bình động bậc1/bậc2 để điều chỉnh bước theo tham số. Chưa có kết quả full-data Adam để kết luận optimizer nào hơn. |
| Vì sao warmup/cosine? | Warmup tăng LR từ thấp đến mức khởi đầu lịch chính; cosine giảm dần LR. Đây là lựa chọn recipe, cần thử nghiệm mới biết tối ưu hay chưa. |
| EMA có phải optimizer khác không? | Không. EMA duy trì trung bình động của trọng số để đánh giá; optimizer vẫn cập nhật model train. Decay lớn giữ ký ức dài hơn. |
| AMP làm gì? | Dùng precision hỗn hợp để giảm bộ nhớ/tăng tốc khi phần cứng hỗ trợ; GradScaler hỗ trợ tránh vấn đề gradient nhỏ. Không đổi số tham số kiến trúc. |
| Gradient clipping làm gì? | Giới hạn norm gradient để giảm cập nhật quá lớn; không phải clipping activation hay tự giảm kênh. |
| Train/val/test khác nhau? | Train để cập nhật weight, val để chọn epoch/recipe, test để đánh giá cuối. Không lấy test làm tiêu chí chọn biến thể tiếp theo. |
| Best và last khác nhau? | Trong selection, best là model có val accuracy cao nhất; last là trạng thái epoch hoàn chỉnh mới nhất để resume. Best không nhất thiết là epoch cuối. |
| Dừng giữa epoch thì sao? | Ctrl+C lần đầu yêu cầu hoàn tất epoch rồi lưu/dừng; lần2 dừng ngay. Khi dừng cứng, epoch chưa hoàn chỉnh phải chạy lại từ last; không có checkpoint từng batch. |
| Resume cần lưu những gì? | Model, EMA, optimizer, scheduler/thông tin lịch, AMP scaler, epoch/step, RNG và lịch sử/config/source/split/environment cần thiết. Chỉ nạp weight không tương đương resume đầy đủ. |
| Sửa model có dùng lại best cũ không? | Không coi đó là resume cùng thí nghiệm. Bài dùng run-name mới, khởi tạo model phù hợp; kết quả cũ chỉ thuộc model cũ. |

### 17.3. Kết quả thật hiện có và phần còn thiếu

| Dataset | Epoch best / weights | Best validation | Test accuracy | Đúng/tổng test | Top-5 test | Macro-F1 test |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 567 / EMA | 97,56% | 97,18% | 9.718/10.000 | 99,87% | 0,971811 |
| CIFAR-100 | 399 / EMA | 83,36% | 82,98% | 8.298/10.000 | 96,75% | 0,829492 |

Nguồn: [BAO_CAO_THEO_DE_FINAL_EXAM.md](BAO_CAO_THEO_DE_FINAL_EXAM.md) và các evidence đã lưu. Cả hai chạy SGD initial LR 0,1, 600 epoch, train 45k/val 5k. Đây là **test của best train 45k**, không gán cho retrain50k hay lượt demo. Tài liệu này không chạy test lại.

Accuracy=`số đúng/tổng×100%`; top-5 là nhãn thật có nằm trong5 lớp có điểm cao nhất không. Với từng lớp, Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2PR/(P+R); Macro-F1 là trung bình F1 của các lớp, xử lý mẫu số0 theo evaluator. Confusion matrix của bài cần đọc đúng trục nhãn thật/dự đoán; xem nhãn trục trên hình, không đoán hướng.

**Chưa hoàn thành phần so sánh SGD/Adam với nhiều initial LR trên cả hai tập.** Bảng kế hoạch hoặc FakeData không thay cho số liệu thực nghiệm. Không nói SGD tốt hơn Adam, kernel 5 tốt hơn kernel 3, hay SE tăng một lượng accuracy cụ thể khi chưa có thử nghiệm tương ứng. Theo quy trình repository, thí nghiệm mới lựa chọn bằng validation, freeze recipe/epoch rồi retrain toàn bộ train và test cuối; số đo archived45k được ghi trung thực theo đúng nguồn.

## 18. Lỗi hay gặp và cách xác định đúng chỗ sửa

| Thông báo/hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `block01.bn_expand` cần80 nhưng nhận96 | Đổi expand mà quên BN | Sửa BN và tiếp tục kiểm tra các nhánh/SE/project phụ thuộc |
| `Block01.concat` | H,W hoặc N hai nhánh khác nhau | Tính công thức của từng nhánh, sửa kernel/padding/stride phù hợp |
| `in_channels must be divisible by groups` | Chọn kênh không chia hết groups | Chọn số kênh/groups hợp lệ; depthwise đổi cả groups theo Cin |
| `Block01.SE` | Gate sai N/E hoặc không phải1×1 | Sửa pooling/SE expand để gate đúng [N,E,1,1] |
| `Block01.ADD` | Main path và skip khác C/H/W | Sửa projection/skip theo thiết kế; không dùng broadcast/reshape tùy tiện |
| `classifier` / lỗi nhân ma trận | Fin không bằng số features sau flatten | Tính C×H×W sau pool, sửa Linear |
| Sửa class mà output không đổi | Đang chạy model instance tạo trước khi sửa | Tạo lại model, chạy lại cell định nghĩa liên quan; tạo lại optimizer khi train mới |
| Bỏ block nhưng P không giảm | Vẫn khởi tạo module không dùng | Bỏ cả khai báo module và lời gọi forward |
| Thêm block nhưng P không tăng | Dùng lại cùng instance | Tạo class/instance độc lập đúng ý đồ |
| Forward được nhưng bảng tổng lỗi | Ledger/sơ đồ/fingerprint còn giả định13 block | Cập nhật các helper tương ứng; kiểm tra lại tổng bằng model thực |
| Resume bị từ chối sau chỉnh sửa | Source/config/model khác run đã lưu | Dùng run-name mới; giữ checkpoint cũ |
| Tưởng code treo vì chưa có log | Train chỉ log sau khi xong cả epoch và ghi đủ artifacts | Chờ epoch hoàn chỉnh; không kết luận dựa trên việc chưa có log batch |
| Model sửa rồi không nạp được archive | Checkpoint thuộc kiến trúc/tên tensor cũ | Trình bày archive bằng bản gốc; dùng run mới cho model mới |

## 19. Bài tự luyện trước buổi báo cáo

1. Trong90 giây, nói được đường đi từ3×32×32 đến10/100 logits và chỉ đúng stem, transition, SE, skip.
2. Tính tay P của stem=1.080, grouped 80→80,g8=7.200 weight, Linear 256→100=25.700; đối chiếu bảng.
3. Giải thích tại sao chunk80 ra 40+40, concat lại80, project trả40 và ADD giữ40.
4. Tự làm lần lượt ba bài độc lập: kernel 5→7; Stage 1 40→48; thêm/bỏ một block Stage 3. Mỗi bài ghi shape dự kiến rồi chạy check.
5. Cố ý đặt context stride 2 để đọc lỗi concat; sau đó thử expand stride 2 và tự sửa skip. Không dùng checkpoint cũ cho bài sửa.
6. Chạy FakeData2 epoch để kiểm tra save/resume, hiểu rằng số accuracy ở đó không được đưa vào kết quả CIFAR.
7. Mở đúng PDF/bảng/số liệu của bản gốc; nói rõ SGD LR 0,1 đã có kết quả, Adam/nhiều LR còn thiếu.
8. Khi được hỏi “vì sao dùng kỹ thuật này?”, trả lời theo cơ chế → động cơ → đánh đổi → bằng chứng hiện có. Nếu chưa có ablation thì nói đúng là chưa đo riêng.

Các tài liệu tra cứu: [giải thích các nn](GIAI_THICH_TAT_CA_NN_TRONG_MODEL.txt), [luồng Conv/forward](LUONG_CONV_FORWARD_CHI_TIET.txt), [công thức shape](CONG_THUC_TINH_SHAPE.md), [đối chiếu đề](DOI_CHIEU_YEU_CAU_FINAL_EXAM.md). File công thức có ví dụ chỉnh sửa từ các phiên làm trước; khi đối chiếu “model hiện tại”, ưu tiên code thực tế và profile mới chạy. Số đo trong tài liệu này ứng với bản gốc40/112/256 và từng biến thể độc lập đã ghi rõ.
