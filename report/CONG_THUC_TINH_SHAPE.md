# Công thức tính shape trong CompactMix

Đọc cùng [compactmix_manual_train.py](compactmix_manual_train.py). Tài liệu lập ngày **15/10/2026**, gồm các lớp, phép toán trong `forward`, và các tensor trong phần train/evaluate.

SHA256 của file Python tại lúc đối chiếu: `f096801aef1272bee7db829e83f6c1a17bb076917a06a41504a8fb9ffcc93ae8`. Nếu sửa code tiếp, tính lại các ví dụ cụ thể bằng công thức tổng quát.

**Shape mô tả kích thước tensor; số tham số học được là một đại lượng khác.** Ví dụ `[N,40,32,32]` là shape, còn `P=1.080` là số trọng số của một Conv cụ thể.

Trong bản code đang chỉnh, `Block01` đã thay đổi stride/kernel/padding và `Block02` đã thay đổi số kênh trung gian SE. [Mục 15](#15-tính-theo-block01-đang-chỉnh) tính theo các giá trị đó. Bảng cấu hình gốc ở mục 14 chỉ dùng để đối chiếu; toàn bộ model đang chỉnh chưa đi qua được SE của `Block01`.

## 1. Ký hiệu cần biết

| Ký hiệu | Ý nghĩa |
|---|---|
| `N` | Số ảnh trong batch hiện tại |
| `C_in`, `C_out` | Số kênh đầu vào, đầu ra |
| `H_in`, `W_in` | Chiều cao, chiều rộng đầu vào |
| `H_out`, `W_out` | Chiều cao, chiều rộng đầu ra |
| `kH`, `kW` | Chiều cao, chiều rộng kernel |
| `sH`, `sW` | Stride theo chiều cao, chiều rộng |
| `pH`, `pW` | Padding ở mỗi phía theo hai chiều |
| `dH`, `dW` | Dilation theo hai chiều |
| `g` | Số nhóm của Conv (`groups`) |
| `K` | Số lớp phân loại: 10 hoặc 100 |
| `D` | Số kênh trung gian trong SE |

Thứ tự tensor ảnh PyTorch trong file:

```text
[N, C, H, W]
 0  1  2  3   <- chỉ số chiều, dùng trong dim=...
```

Ví dụ batch đầy đủ mặc định:

```text
Ảnh CIFAR: [128, 3, 32, 32]
N=128; C=3; H=32; W=32
```

`kernel_size=3` nghĩa là `kH=kW=3`. `kernel_size=(3,5)` nghĩa là `kH=3,kW=5`. Tương tự với stride, padding, dilation.

### Floor và ceil

- `floor(x)`: số nguyên lớn nhất không vượt quá `x`.
- `ceil(x)`: số nguyên nhỏ nhất không nhỏ hơn `x`.

```text
floor(1.5)=1; floor(1.9)=1; floor(-1.5)=-2
ceil(1.5)=2; ceil(2.0)=2
```

Trong công thức Conv, tính phần bên trong rồi mới `floor`; không làm tròn thông thường.

## 2. `nn.Conv2d`: công thức tổng quát

Đầu vào:

```text
X: [N, C_in, H_in, W_in]
```

Đầu ra:

$$
H_{out}=\left\lfloor\frac{H_{in}+2p_H-d_H(k_H-1)-1}{s_H}+1\right\rfloor
$$

$$
W_{out}=\left\lfloor\frac{W_{in}+2p_W-d_W(k_W-1)-1}{s_W}+1\right\rfloor
$$

```text
Y: [N, out_channels, H_out, W_out]
```

Viết dạng dễ thế số:

```text
H_out = floor((H_in + 2*pH - dH*(kH-1) - 1)/sH + 1)
W_out = floor((W_in + 2*pW - dW*(kW-1) - 1)/sW + 1)
C_out = out_channels
N_out = N_in
```

### Điều kiện hợp lệ

- `C` của input phải bằng `in_channels` của Conv.
- `in_channels` và `out_channels` đều chia hết cho `groups`.
- Kernel, stride, dilation và số kênh phải dương; padding không âm.
- `H_out,W_out` phải dương; kernel hiệu dụng phải lọt trong input sau padding.

`bias`, `groups` không trực tiếp xuất hiện trong công thức H/W. `groups` thay đổi cách nối kênh và số trọng số, còn `C_out` vẫn lấy từ `out_channels`.

### Cách nhớ qua kernel hiệu dụng

```text
kH_eff = dH*(kH-1)+1
kW_eff = dW*(kW-1)+1

H_out = floor((H_in + 2*pH - kH_eff)/sH)+1
W_out = floor((W_in + 2*pW - kW_eff)/sW)+1
```

### Khi nào Conv giữ H, W?

Với `stride=1`, padding đối xứng giữ nguyên kích thước nếu:

$$
2p=d(k-1)
$$

| Kernel | Dilation | Stride | Padding | Kết quả |
|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 0 | Giữ H,W |
| 3 | 1 | 1 | 1 | Giữ H,W |
| 5 | 1 | 1 | 2 | Giữ H,W |
| 7 | 1 | 1 | 3 | Giữ H,W |
| 3 | 2 | 1 | 2 | Giữ H,W; kernel hiệu dụng là 5 |

Điều kiện áp dụng riêng cho mỗi chiều. Nếu `d*(k-1)` lẻ, không có padding nguyên đối xứng thỏa công thức này khi stride=1; có thể cần cách padding khác nếu muốn giữ kích thước.

### Ví dụ Conv đầu tiên `self.stem`

```python
nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=1,
          groups=1, bias=False)  # dilation mặc định=1
```

```text
H_out = floor((32 + 2*1 - 1*(3-1) - 1)/1 + 1)
      = floor(31+1) = 32
W_out = 32

[N,3,32,32] -> [N,40,32,32]
```

### Conv pointwise, depthwise, grouped

Ba loại này đều dùng cùng công thức H/W.

| Loại | Cấu hình ví dụ | Input → output |
|---|---|---|
| Pointwise | `40→80,k=1,s=1,p=0,g=1` | `[N,40,H,W] → [N,80,H,W]` |
| Depthwise | `40→40,k=3,s=1,p=1,g=40` | `[N,40,H,W] → [N,40,H,W]` |
| Grouped | `80→80,k=3,s=1,p=1,g=8` | `[N,80,H,W] → [N,80,H,W]` |

Depthwise trong model dùng `C_in=C_out=groups`. Đó là cấu hình hiện dùng; PyTorch cũng cho phép depthwise có `C_out` là bội của `C_in`.

## 3. `nn.BatchNorm2d`: giữ shape

$$
[N,C,H,W]\longrightarrow[N,C,H,W]
$$

Điều kiện: `num_features=C` của input.

```text
BatchNorm2d(80): [N,80,32,32] -> [N,80,32,32]
```

- `eps`, `momentum`, gamma, beta thay đổi giá trị hoặc cách cập nhật thống kê; không thay đổi shape.
- Trong chế độ train thông thường, cần hơn một giá trị trên mỗi kênh để tính thống kê: `N*H*W > 1`. Ví dụ `[1,C,1,1]` sẽ lỗi khi BN dùng thống kê batch.
- Ở `eval()` với running stats như cấu hình mặc định, BN dùng thống kê đã lưu.

## 4. `nn.SiLU`, `nn.Sigmoid`, `nn.Dropout`: giữ shape

Các lớp này xử lý từng phần tử:

$$
\operatorname{shape}(f(X))=\operatorname{shape}(X)
$$

| Lớp | Công thức giá trị hoặc hoạt động | Shape |
|---|---|---|
| `SiLU` | `x*sigmoid(x)` | Giữ nguyên |
| `Sigmoid` | `1/(1+exp(-x))` | Giữ nguyên |
| `Dropout(p)` | Train: bỏ ngẫu nhiên phần tử và chia phần giữ lại cho `1-p`; eval: giữ input | Giữ nguyên |

Quy tắc chia của Dropout áp dụng với `0 <= p < 1`; nếu tự đặt `p=1`, PyTorch đặt toàn bộ output về 0 khi train. Code hiện dùng `p=0`.

Ví dụ:

```text
SiLU:        [N,80,32,32] -> [N,80,32,32]
Sigmoid SE:  [N,80,1,1]   -> [N,80,1,1]
Dropout head:[N,256]      -> [N,256]
```

Đổi `p`, hoặc đổi SiLU sang một activation xử lý từng phần tử khác, không tự thay đổi C/H/W.

## 5. `DropPath`: giữ shape

DropPath là class tự viết kế thừa `nn.Module`:

```text
input:  X [N,C,H,W]
mask:     [N,1,1,1]
output: Y [N,C,H,W]
```

Khi train với `0 <= p < 1`:

$$
Y_{n,c,h,w}=X_{n,c,h,w}\frac{m_n}{1-p},\quad m_n\in\{0,1\}
$$

Mask được broadcast theo C/H/W. Có thể bỏ cả nhánh residual cho một ảnh, nhưng số phần tử và shape không đổi. Khi eval hoặc `p=0`, `Y=X`.

## 6. `nn.AvgPool2d`: tính H/W theo cửa sổ

Với cấu hình `ceil_mode=False` đang dùng:

$$
H_{out}=\left\lfloor\frac{H_{in}+2p_H-k_H}{s_H}+1\right\rfloor
$$

$$
W_{out}=\left\lfloor\frac{W_{in}+2p_W-k_W}{s_W}+1\right\rfloor
$$

```text
[N,C,H_in,W_in] -> [N,C,H_out,W_out]
```

AvgPool2d không có dilation. Nếu không ghi stride, stride mặc định bằng kernel_size. Padding pooling còn phải nằm trong giới hạn API; công thức hình học không thay thế việc kiểm tra tham số hợp lệ.

Hai pooling thực tế:

```text
AvgPool2d(k=2,s=2,p=0)

H_out = floor((H_in-2)/2+1) = floor(H_in/2)
W_out = floor(W_in/2)

t1_pool: [N,40,32,32]  -> [N,40,16,16]
t2_pool: [N,112,16,16] -> [N,112,8,8]
```

Với cạnh lẻ, ví dụ `H_in=31`: `H_out=floor(31/2)=15`.

`count_include_pad` và `divisor_override` ảnh hưởng giá trị trung bình, không đổi shape khi các thông số hình học khác giữ nguyên.

Nếu tự đổi sang `ceil_mode=True`, tính ứng viên bằng `ceil` thay vì `floor`. Sau đó, nếu `(H_out-1)*sH >= H_in+pH` thì giảm `H_out` đi 1, vì không lấy cửa sổ bắt đầu hoàn toàn trong phần padding bên phải. Làm tương tự với W.

## 7. `nn.AdaptiveAvgPool2d`: shape theo output_size

Với `output_size=(h,w)`:

$$
[N,C,H_{in},W_{in}]\longrightarrow[N,C,h,w]
$$

- `output_size=1`: tương đương `(1,1)`.
- `output_size=2`: tương đương `(2,2)`.
- `output_size=(2,3)`: đầu ra H=2,W=3.
- `None` ở một chiều nghĩa là giữ kích thước input của chiều đó.

Ví dụ:

```text
AdaptiveAvgPool2d(1):      [N,80,32,32] -> [N,80,1,1]
AdaptiveAvgPool2d(1):      [N,256,8,8]  -> [N,256,1,1]
AdaptiveAvgPool2d((2,3)):  [N,256,8,8]  -> [N,256,2,3]
AdaptiveAvgPool2d((None,1)):[N,80,H,W]  -> [N,80,H,1]
```

**AdaptiveAvgPool2d giữ nguyên C.** Đổi `80→8` trong SE do Conv `se_reduce`, không phải do pooling.

Nếu cần tính vùng trung bình cho ô đầu ra `(i,j)`:

```text
h_start = floor(i*H_in/h)
h_end   = ceil((i+1)*H_in/h)
w_start = floor(j*W_in/w)
w_end   = ceil((j+1)*W_in/w)

Y[n,c,i,j] = mean(X[n,c,h_start:h_end,w_start:w_end])
```

Với `h=w=1`, lấy trung bình toàn bộ H/W của từng ảnh, từng kênh.

## 8. `y.chunk(2, dim=1)`: chia số kênh

Đầu vào `[N,C,H,W]`. Khi `C` chẵn và dương:

$$
A:[N,C/2,H,W],\quad B:[N,C/2,H,W]
$$

```text
[N,80,H,W]  -> [N,40,H,W]  và [N,40,H,W]
[N,224,H,W] -> [N,112,H,W] và [N,112,H,W]
[N,512,H,W] -> [N,256,H,W] và [N,256,H,W]
```

Khi `C>=2` lẻ:

```text
C_a=ceil(C/2)
C_b=floor(C/2)

C=81 -> 41 kênh và 40 kênh
```

Hai Conv nhận hai nhánh phải có `in_channels` khớp từng phần. Nếu `C=1`, `chunk(2)` chỉ trả một phần, nên phép gán `a,b=...` lỗi.

Tổng quát, khi chia chiều có độ dài dương `L` thành `q` chunk:

```text
t = ceil(L/q)                # kích thước chunk, trừ phần cuối có thể nhỏ hơn
số chunk thực tế = ceil(L/t) # có thể ít hơn q
```

Đây là chia dữ liệu, không phải giảm giá trị bằng pooling. Tensor `y` ban đầu vẫn giữ shape cũ.

## 9. `torch.cat(..., dim=1)` / `concat_checked`

Với:

```text
A: [N,C_a,H,W]
B: [N,C_b,H,W]
```

Ta có:

$$
\operatorname{cat}(A,B,\mathrm{dim}=1):[N,C_a+C_b,H,W]
$$

Điều kiện trong trường hợp tensor 4 chiều của model:

```text
N_a=N_b; H_a=H_b; W_a=W_b
```

Số kênh có thể khác nhau. Các tensor cần tương thích về kiểu dữ liệu và cùng thiết bị; trong file, các nhánh được tạo từ cùng tensor.

Ví dụ:

```text
[N,40,32,32] cat [N,40,32,32] -> [N,80,32,32]
[N,40,32,32] cat [N,48,32,32] -> [N,88,32,32]
[N,40,32,32] cat [N,40,30,30] -> LỖI H,W không bằng nhau
```

BN/Conv nhận kết quả phải nhận tổng số kênh mới. Tổng quát, concat theo chiều nào thì cộng độ dài chiều đó; các chiều còn lại phải khớp.

## 10. `feature * gate` / `multiply_checked`

Hàm trong file yêu cầu chính xác:

```text
feature: [N,C,H,W]
gate:    [N,C,1,1]
```

Kết quả:

$$
Y:[N,C,H,W],\quad Y_{n,c,h,w}=F_{n,c,h,w}G_{n,c,0,0}
$$

Ví dụ:

```text
[128,80,32,32] * [128,80,1,1] -> [128,80,32,32]
```

Đây là nhân từng phần tử có broadcasting; không phải nhân ma trận.

### Quy tắc broadcasting tổng quát của PyTorch

So các chiều từ phải sang trái; thiếu chiều bên trái thì xem như có độ dài 1. Mỗi cặp chiều phải bằng nhau, hoặc một chiều bằng 1. Khi hợp lệ, chiều kết quả lấy độ dài không phải 1, hoặc 1 nếu cả hai đều 1, trong trường hợp kích thước dương đang xét.

```text
[N,C,H,W]
[N,C,1,1]
---------
[N,C,H,W]
```

**Điều kiện của `multiply_checked` chặt hơn broadcasting tổng quát.** Ví dụ gate `[1,C,1,1]` có thể nhân bằng PyTorch với feature có N>1, nhưng hàm này vẫn từ chối vì yêu cầu N gate bằng N feature.

## 11. `x + residual` / `residual_checked`

Trong file, hai shape phải giống hoàn toàn:

$$
\operatorname{shape}(X)=\operatorname{shape}(R)=[N,C,H,W]
$$

$$
Y=X+R\quad\Rightarrow\quad\operatorname{shape}(Y)=[N,C,H,W]
$$

```text
[N,40,32,32] + [N,40,32,32] -> [N,40,32,32]
[N,40,32,32] + [N,48,32,32] -> LỖI khác C
[N,40,32,32] + [N,40,3,3]   -> LỖI khác H,W
```

**ADD không cộng số kênh.** Concat 40 và 40 kênh ra 80; ADD hai tensor 40 kênh ra 40 kênh.

Phép `+` gốc của PyTorch có thể broadcast, nhưng `residual_checked` chủ động yêu cầu shape bằng nhau để phát hiện nối residual sai.

## 12. `x.flatten(1)`, `reshape` và `nn.Linear`

### Flatten

$$
[N,C,H,W]\xrightarrow{\mathrm{flatten}(1)}[N,C\cdot H\cdot W]
$$

```text
[N,256,1,1] -> [N,256]
[N,256,2,2] -> [N,1024]
[N,256,8,8] -> [N,16384]
```

`flatten(1)` giữ chiều batch. Tổng quát, gộp các chiều `start_dim..end_dim` bằng tích các kích thước đó; các chiều ngoài dải giữ nguyên.

### Reshape

Tổng số phần tử phải bảo toàn:

$$
\prod_i a_i=\prod_j b_j
$$

Ví dụ trong confusion matrix: vector `[K*K]` được `reshape(K,K)` thành `[K,K]`. Nếu một chiều dùng `-1`, PyTorch suy ra chiều đó từ tổng số phần tử; các kích thước vẫn phải hợp lệ.

### Linear

```python
nn.Linear(in_features=F_in, out_features=F_out)
```

$$
[N,F_{in}]\longrightarrow[N,F_{out}]
$$

Điều kiện: chiều cuối input bằng `in_features`. Tổng quát, Linear giữ các chiều phía trước và thay chiều cuối:

```text
[d0,d1,...,F_in] -> [d0,d1,...,F_out]
```

Trong head:

```text
[N,256,8,8]
    -> AdaptiveAvgPool2d(1) -> [N,256,1,1]
    -> flatten(1)          -> [N,256]
    -> Dropout             -> [N,256]
    -> Linear(256,K)       -> [N,K]
```

Nếu đổi pooling đầu ra thành `(h,w)` và kênh cuối thành C:

$$
\text{classifier.in_features}=C\cdot h\cdot w
$$

`out_features=K` gắn với số lớp của dataset; không suy ra K từ H/W.

## 13. Công thức chung của một MixBlock

Ký hiệu đầu vào `[N,C,H,W]`, số kênh sau expand là E, số kênh ẩn SE là D.

Bảng sau giả sử Conv giữ H/W như cấu hình gốc, E chẵn, project xuất lại C và gate SE giữ 1x1:

| Bước trong forward | Shape |
|---|---|
| Input, đồng thời là skip | `[N,C,H,W]` |
| `bn_pre → act_pre` | `[N,C,H,W]` |
| `expand` Conv 1x1 C→E | `[N,E,H,W]` |
| `bn_expand → act_expand` | `[N,E,H,W]` |
| `chunk(2,dim=1)` | Hai tensor `[N,E/2,H,W]` |
| `local`, `context` | Mỗi nhánh `[N,E/2,H,W]` |
| `concat_checked` | `[N,E,H,W]` |
| `bn_mix → act_mix` | `[N,E,H,W]` |
| `grouped` Conv E→E | `[N,E,H,W]` |
| `bn_gate → act_gate`; lưu `feature` | `[N,E,H,W]` |
| `se_pool` | `[N,E,1,1]` |
| `se_reduce` Conv E→D | `[N,D,1,1]` |
| `se_act` | `[N,D,1,1]` |
| `se_expand` Conv D→E | `[N,E,1,1]` |
| `se_sigmoid` | `[N,E,1,1]` |
| `multiply_checked(feature,gate)` | `[N,E,H,W]` |
| `project` Conv E→C | `[N,C,H,W]` |
| `drop` | `[N,C,H,W]` |
| `residual_checked(x,y)` | `[N,C,H,W]` |

Nếu đổi stride/kernel/padding, phải tính H/W tại từng Conv bằng mục 2. Cần kiểm tra cả hai nhánh trước concat, kích thước gate trước multiply và kích thước skip trước ADD.

## 14. Bảng shape của cấu hình gốc để đối chiếu

**Đây là cấu hình trước các chỉnh sửa đang có trong Block01/Block02.** Dùng bảng này để hiểu các stage; không coi là kết quả forward của toàn bộ code đang chỉnh.

| Vị trí | Shape input | Shape output |
|---|---|---|
| Stem Conv3x3 3→40 | `[N,3,32,32]` | `[N,40,32,32]` |
| Block01, Block02 | `[N,40,32,32]` | `[N,40,32,32]` |
| Transition1 BN, SiLU | `[N,40,32,32]` | `[N,40,32,32]` |
| Transition1 AvgPool2d(2,2) | `[N,40,32,32]` | `[N,40,16,16]` |
| Transition1 Conv1x1 40→112 | `[N,40,16,16]` | `[N,112,16,16]` |
| Block03,04,05 | `[N,112,16,16]` | `[N,112,16,16]` |
| Transition2 BN, SiLU | `[N,112,16,16]` | `[N,112,16,16]` |
| Transition2 AvgPool2d(2,2) | `[N,112,16,16]` | `[N,112,8,8]` |
| Transition2 Conv1x1 112→256 | `[N,112,8,8]` | `[N,256,8,8]` |
| Block06..13 | `[N,256,8,8]` | `[N,256,8,8]` |
| Head BN, SiLU | `[N,256,8,8]` | `[N,256,8,8]` |
| Head AdaptiveAvgPool2d(1) | `[N,256,8,8]` | `[N,256,1,1]` |
| `flatten(1)` | `[N,256,1,1]` | `[N,256]` |
| Head Dropout | `[N,256]` | `[N,256]` |
| Linear CIFAR-10 | `[N,256]` | `[N,10]` |
| Linear CIFAR-100 | `[N,256]` | `[N,100]` |

Các số bên trong block gốc, áp dụng bảng mục 13:

| Block | C | E | Mỗi nhánh sau chunk | D của SE | H,W |
|---|---:|---:|---:|---:|---|
| 01,02 | 40 | 80 | 40 | 8 | 32,32 |
| 03,04,05 | 112 | 224 | 112 | 14 | 16,16 |
| 06..13 | 256 | 512 | 256 | 32 | 8,8 |

## 15. Tính theo Block01 đang chỉnh

Các giá trị đọc trực tiếp từ file tại lúc lập tài liệu:

```text
expand:    40→80, k=1,s=3,p=0,d=1
local:     40→40, k=5,s=3,p=1,d=1,g=40
context:   40→40, k=3,s=3,p=0,d=1,g=40
grouped:   80→80, k=2,s=3,p=3,d=1,g=8
se_pool:   output_size=1
se_reduce: 80→8,  k=2,s=1,p=2,d=1
se_expand: 8→80,  k=1,s=1,p=3,d=1
project:   80→40, k=1,s=1,p=0,d=1
```

### Tính H và W từng bước

Input sau stem là `[N,40,32,32]`. H=W nên chỉ cần thế số cho một cạnh; cạnh còn lại tính giống hệt.

```text
expand:
  floor((32+0-1*(1-1)-1)/3+1)
  = floor(31/3+1) = 11

local, nhận cạnh 11:
  floor((11+2*1-1*(5-1)-1)/3+1)
  = floor(8/3+1) = 3

context, nhận cạnh 11:
  floor((11+0-1*(3-1)-1)/3+1)
  = floor(8/3+1) = 3

grouped, nhận cạnh 3 sau concat:
  floor((3+2*3-1*(2-1)-1)/3+1)
  = floor(7/3+1) = 3

se_pool:
  cạnh được yêu cầu = 1

se_reduce, nhận cạnh 1:
  floor((1+2*2-1*(2-1)-1)/1+1) = 4

se_expand, nhận cạnh 4:
  floor((4+2*3-1*(1-1)-1)/1+1) = 10
```

| Bước | Shape hiện tại |
|---|---|
| Input/skip | `[N,40,32,32]` |
| BN, SiLU trước expand | `[N,40,32,32]` |
| Expand, BN, SiLU | `[N,80,11,11]` |
| Chunk | Hai nhánh `[N,40,11,11]` |
| Local và context | Mỗi nhánh `[N,40,3,3]` |
| Concat, BN, SiLU | `[N,80,3,3]` |
| Grouped, BN, SiLU; feature | `[N,80,3,3]` |
| SE pool | `[N,80,1,1]` |
| SE reduce, SiLU | `[N,8,4,4]` |
| SE expand, Sigmoid; gate | `[N,80,10,10]` |
| Multiply | **Dừng: gate cần `[N,80,1,1]`** |

Cả local và context hiện ra 3x3 nên concat hợp lệ. Lỗi kết nối đầu tiên xảy ra tại `Block01.SE`: gate có H/W=10, còn hàm đòi H/W=1.

Nếu chỉ chỉnh nhánh SE để gate trở về `[N,80,1,1]`, feature sau multiply vẫn là `[N,80,3,3]`. Project hiện giữ cạnh 3, tạo `[N,40,3,3]`; DropPath giữ nguyên. ADD sau đó vẫn không khớp skip `[N,40,32,32]`.

Vì vậy cần tính cả hai ràng buộc độc lập: **gate SE có kích thước đúng** và **nhánh residual khớp skip trước ADD**. Tài liệu chỉ chỉ ra phép tính; các thông số code của bạn được giữ nguyên.

### Block02 đang chỉnh: thay SE 80→8→80 bằng 80→80→80

Nếu đưa riêng input phù hợp `[N,40,32,32]` vào Block02, các bước ngoài SE vẫn giữ H/W=32. Nhánh SE hiện là:

```text
feature [N,80,32,32]
  -> pool        [N,80,1,1]
  -> Conv80→80   [N,80,1,1]
  -> SiLU        [N,80,1,1]
  -> Conv80→80   [N,80,1,1]
  -> Sigmoid     [N,80,1,1]
```

Thay đổi này vẫn khớp shape để multiply, project và ADD; tuy nhiên làm tăng số tham số và phép tính trong SE. Nó không sửa lỗi kết nối ở Block01 trước đó.

## 16. Các lớp chỉ xuất hiện trong profiler

`inspect_model` nhắc tới các kiểu sau bằng `isinstance`. Chúng chưa được tạo thành layer trong `CompactMixManual` hiện tại, nhưng công thức hữu ích nếu bạn thay lớp.

| Lớp | Công thức shape | Điều kiện chính |
|---|---|---|
| `nn.GroupNorm(G,C)` | `[N,C,H,W] → [N,C,H,W]` | C input bằng num_channels, C chia hết cho G |
| `nn.LayerNorm(normalized_shape)` | Giữ nguyên | Các chiều cuối input khớp normalized_shape |
| `nn.ReLU`, `nn.GELU`, `nn.LeakyReLU` | Giữ nguyên | Activation từng phần tử |
| `nn.Identity` | Input → chính input | Không biến đổi |
| `nn.Flatten(start_dim=1)` | `[N,C,H,W] → [N,C*H*W]` | Như `x.flatten(1)` |
| `nn.AdaptiveMaxPool2d((h,w))` | `[N,C,H,W] → [N,C,h,w]` | Khi `return_indices=False` như mặc định |
| `nn.MaxPool2d` | Dùng công thức bên dưới | Kiểm tra kernel/stride/padding/dilation hợp lệ |

Với MaxPool2d, `ceil_mode=False`:

```text
H_out = floor((H_in+2*pH-dH*(kH-1)-1)/sH+1)
W_out = floor((W_in+2*pW-dW*(kW-1)-1)/sW+1)
```

Nếu `ceil_mode=True`, dùng ceil và loại cửa sổ bắt đầu hoàn toàn ở padding bên phải theo điều kiện ở mục 6. Nếu pooling trả cả indices, mỗi tensor indices có cùng shape với tensor giá trị output.

## 17. Shape dữ liệu, augmentation, batch và Mixup/CutMix

### Dữ liệu đầu vào

| Bước | Shape/kích thước |
|---|---|
| Một ảnh CIFAR dạng PIL | Rộng 32, cao 32, 3 kênh |
| `RandomCrop(32,padding=4)` | Đệm tạm thành 40x40, rồi crop lại 32x32 |
| `RandomHorizontalFlip` | Giữ chiều cao/rộng |
| AutoAugment/RandAugment/TrivialAugment đang gọi | Giữ kích thước ảnh 32x32 |
| `ToTensor()` | Một ảnh thành `[3,32,32]` |
| `Normalize(mean,std)` | `[3,32,32] → [3,32,32]` |
| `RandomErasing`, nếu bật | Giữ shape, thay giá trị một vùng |
| DataLoader ghép N ảnh | `images:[N,3,32,32]`, `labels:[N]` |

DataLoader hiện không bật `drop_last`. Nếu có M ảnh, batch size B:

$$
\text{số batch}=\left\lceil M/B\right\rceil
$$

```text
Batch thứ j (đánh số từ 0): N_j=min(B,M-j*B)
```

Ví dụ B=128:

| Tập | M | Số batch | N của batch cuối |
|---|---:|---:|---:|
| Train selection | 45.000 | 352 | 72 |
| Validation | 5.000 | 40 | 8 |
| Train final | 50.000 | 391 | 80 |
| Test | 10.000 | 79 | 16 |

### `mix_batch`

```text
images:                    [N,3,32,32]
labels:                    [N]
permutation=randperm(N):    [N]
paired=labels[permutation]: [N]
targets=one_hot(labels,K):  [N,K]
```

- Mixup: `lambda*images+(1-lambda)*images[permutation]` giữ `[N,3,32,32]`.
- CutMix: cắt/dán vùng trên cùng tensor ảnh, giữ `[N,3,32,32]`.
- Trộn targets và label smoothing giữ `[N,K]`.
- `lambda`/`coefficient` là một số vô hướng.

## 18. Shape loss, dự đoán và số liệu đánh giá

### Train với nhãn mềm

```python
loss = -(targets * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
```

| Bước | Shape |
|---|---|
| `logits`, `targets` | `[N,K]` |
| `logits.float()` | `[N,K]` |
| `F.log_softmax(...,dim=1)` | `[N,K]` |
| `targets * log_probabilities` | `[N,K]` |
| `.sum(1)` | `[N]` |
| `.mean()` | `[]`: tensor vô hướng |

### `F.cross_entropy`

Với input `[N,K]`, nhãn nguyên `[N]` như code đánh giá:

```text
reduction='none': [N]
reduction='sum':  []
reduction='mean': []  # mặc định, cũng dùng trong kiểm tra backward
```

### Dự đoán và top-k

```text
logits.argmax(1):        [N,K] -> [N]
predictions.eq(labels): [N]   -> [N] kiểu bool
.sum():                 [N]   -> []
.item():                []    -> một số Python

k=min(5,K)
logits.topk(k,dim=1).values:  [N,k]
logits.topk(k,dim=1).indices: [N,k]
labels[:,None]:              [N,1]
indices.eq(labels[:,None]):  [N,k]  # broadcast [N,1]
.any(1):                    [N]
.sum():                     []
```

### Confusion matrix, precision, recall, F1

```text
labels*K+predictions:                     [N]
torch.bincount(...,minlength=K*K):        [K*K]
.reshape(K,K):                           [K,K]
confusion.diag():                        [K]
confusion.sum(1), confusion.sum(0):       [K]
precision, recall, f1:                   [K]
f1.mean():                               []
```

Vector bincount ở đây dài đúng K*K vì labels và predictions đều trong `[0,K-1]`. Hàng confusion là lớp thật, cột là lớp dự đoán.

### Quy tắc reduction tổng quát

- `sum(dim=d)`, `mean(dim=d)`, `argmax(dim=d)`, `any(dim=d)`, `all(dim=d)` với `keepdim=False` bỏ chiều d.
- Với `keepdim=True`, chiều d còn lại nhưng có độ dài 1.
- `sum()`/`mean()` không chỉ định dim giảm toàn bộ thành tensor vô hướng.
- `clamp_min`, `isfinite`, `.float`, `.long` giữ shape. `.eq` giữa hai tensor cùng shape cũng giữ shape; khi khác shape thì áp dụng broadcasting nếu hợp lệ.

## 19. Shape weight, gradient, EMA và thao tác quản lý

### Shape tham số

```text
Conv2d.weight: [C_out, C_in/groups, kH, kW]
Conv2d.bias:   [C_out], nếu bias=True

BatchNorm2d.weight (gamma): [C]
BatchNorm2d.bias (beta):    [C]
running_mean, running_var: [C]
num_batches_tracked:       []

Linear.weight: [F_out,F_in]
Linear.bias:   [F_out], nếu bias=True
```

Nếu một tham số W có gradient sau backward thì `W.grad` có cùng shape với W. Input gradient, khi được yêu cầu tính, cũng có shape như input tương ứng.

### Các thao tác giữ shape

| Hàm/thao tác trong file | Tác động shape |
|---|---|
| `nn.init.kaiming_normal_`, `trunc_normal_`, `zeros_` | Gán giá trị, giữ shape tensor được truyền vào |
| `clip_grad_norm_` | Giữ shape mỗi gradient; trả norm là tensor vô hướng |
| `zero_grad(set_to_none=True)` | Đặt gradient thành None, không đổi shape tham số |
| `optimizer.step()` | Cập nhật giá trị tham số, giữ shape |
| `scaler.scale(loss)` | Loss vẫn là scalar; scale không đổi shape |
| `scaler.unscale_`, `step`, `update` | Không thiết kế lại shape model |
| `value.lerp_(current[name],...)` trong EMA | Giữ shape; tensor EMA và model cùng tên có shape tương ứng |
| `value.copy_(...)` cho buffer EMA | Giữ shape đích |
| `.clone()`, `.detach()`, `.cpu()`, `.to(device)` | Giữ kích thước; có thể đổi nơi lưu/thiết bị/quan hệ gradient |
| `.float()` và autocast | Có thể đổi dtype, không đổi kích thước |
| `.train()`, `.eval()`, `no_grad()` | Đổi chế độ tính toán, không thay kiến trúc |
| `numel()` | Trả số phần tử `N*C*H*W` của tensor ảnh |
| `nn.Module`/`forward` | Shape tổng hợp do các bước bên trong quyết định |

Các hàm lưu checkpoint/JSON/CSV, logger, khóa run, RNG, fingerprint, parser, scheduler và quản lý đường dẫn không phải phép biến đổi đặc trưng `[N,C,H,W]`. Không có công thức shape layer riêng cho chúng. `make_model` dựng kiến trúc; `inspect_model`/`input_checks` quan sát và kiểm tra shape, không tự sửa.

## 20. Phân biệt công thức shape và số tham số

Conv:

$$
P=C_{out}\left(\frac{C_{in}}{groups}\right)k_Hk_W
+\begin{cases}C_{out},&bias=True\\0,&bias=False\end{cases}
$$

BatchNorm affine mặc định: `P=2*C` (gamma và beta).

Linear: `P=F_out*F_in + F_out` khi có bias.

Ví dụ stem `3→40,k=3,g=1,bias=False`:

```text
Shape output = [N,40,32,32]
Số tham số P = 40*(3/1)*3*3 = 1.080
```

N, H_out, W_out không xuất hiện trong số trọng số Conv; cùng bộ trọng số được dùng trên nhiều ảnh/vị trí. Tăng stride có thể giảm H/W nhưng không tự làm giảm số trọng số nếu C/kernel/groups giữ nguyên.

## 21. Bảng tra nhanh khi sửa code

| Bạn sửa | Cần tự tính/kiểm tra lại |
|---|---|
| Conv `out_channels` | BN sau Conv, chunk, input Conv kế tiếp, concat, SE, projection và skip |
| Conv `kernel/stride/padding/dilation` | H/W bằng công thức Conv, rồi kiểm tra concat/multiply/ADD |
| Conv `groups` | C_in,C_out chia hết cho groups; H/W không đổi nếu hình học giữ nguyên |
| `chunk` hoặc số kênh trước chunk | Số phần thực tế, số kênh từng nhánh và in_channels của nhánh |
| Nhánh local/context | N/H/W trước concat; tổng C sau concat |
| Pooling SE | Gate cuối cần đúng `[N,C_feature,1,1]` |
| SE hidden D | Reduce xuất D; expand nhận D; expand xuất C_feature |
| Project | Output phải khớp skip cả N/C/H/W |
| Head pool hoặc kênh cuối | `Linear.in_features=C*h*w` |
| BN eps/momentum, Dropout p, DropPath p | Không đổi shape, nhưng thay hành vi train |

Sau khi tự sửa, kiểm tra bằng đúng Python của bạn:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 report/compactmix_manual_train.py --check --dataset both
```

Với bản đang chỉnh được mô tả ở mục 15, lệnh sẽ báo lỗi tại SE của Block01. Đây là kiểm tra kết nối, chưa phải train hoặc đo accuracy. Khi mọi shape đã hợp lệ và muốn train kiến trúc đã sửa, dùng `--run-name` mới để giữ checkpoint của source cũ.

Tham khảo API: [torch.nn](https://docs.pytorch.org/docs/stable/nn.html), [torch.chunk](https://docs.pytorch.org/docs/stable/generated/torch.chunk.html), [broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html).
