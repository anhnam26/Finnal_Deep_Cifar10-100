"""Inline code for measured block-wise CompactMix parameters/MACs tables."""

ACCOUNTING_CELL = r'''
def cumulative_model_accounting(model,num_classes):
    # Chạy tensor thật; đếm mỗi leaf đúng một lần, không đếm lại container.
    rows=[]
    names=["stem","block01","block02","transition1","block03","block04","block05",
           "transition2",*[f"block{i:02}" for i in range(6,14)],"head","classifier"]
    ledger={name:{"block":name,"input_shape":None,"output_shape":None,
                  "parameters":0,"macs":0,"flops":0,"leaves":[]} for name in names}
    handles=[]
    mode=model.training

    def group(name):
        root=name.split(".")[0]
        if root.startswith("t1_"):return "transition1"
        if root.startswith("t2_"):return "transition2"
        if root.startswith("head_"):return "head"
        return root if root in ledger else None

    def observe(name):
        def hook(layer,inputs,output):
            key=group(name)
            if key is None:return
            row=ledger[key]
            if row["input_shape"] is None:row["input_shape"]=list(inputs[0].shape)
            row["output_shape"]=list(output.shape)
            n=output.numel(); macs=flops=params=0
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                cost=(layer.in_features if isinstance(layer,nn.Linear) else
                      layer.in_channels//layer.groups*math.prod(layer.kernel_size))
                macs=n*cost; flops=2*macs+(n if layer.bias is not None else 0)
            elif isinstance(layer,nn.BatchNorm2d):flops=4*n
            elif isinstance(layer,(nn.GroupNorm,nn.LayerNorm)):flops=8*n
            elif isinstance(layer,(nn.SiLU,nn.ReLU,nn.GELU,nn.Sigmoid,nn.LeakyReLU)):
                flops=n*(6 if isinstance(layer,nn.SiLU) else 8 if isinstance(layer,nn.GELU)
                         else 4 if isinstance(layer,nn.Sigmoid) else 2 if isinstance(layer,nn.LeakyReLU) else 1)
            elif isinstance(layer,(nn.AvgPool2d,nn.MaxPool2d,nn.AdaptiveAvgPool2d,nn.AdaptiveMaxPool2d)):
                flops=inputs[0].numel()+n
            elif isinstance(layer,ManualBlock):
                flops=n+layer._gate_elements  # residual ADD và SE MUL
            if not list(layer.children()):
                params=sum(p.numel() for p in layer.parameters(recurse=False) if p.requires_grad)
                row["leaves"].append({"name":name,"input_shape":list(inputs[0].shape),
                    "output_shape":list(output.shape),"parameters":params,"macs":macs,"flops":flops})
            row["parameters"]+=params;row["macs"]+=macs;row["flops"]+=flops
        return hook

    try:
        model.cpu().eval()
        for name,layer in model.named_modules():
            if name:handles.append(layer.register_forward_hook(observe(name)))
        profile=inspect_model(model,num_classes,backward=False)
    finally:
        for handle in handles:handle.remove()
        model.train(mode)

    def shape_text(shape):
        return "×".join(str(v) for v in shape[1:])  # bỏ batch N=1

    cumulative={"parameters":0,"macs":0,"flops":0}
    for name in names:
        row=ledger[name]
        if row["input_shape"] is None:raise ValueError("Khối chưa chạy trong forward: "+name)
        row["input"]=shape_text(row["input_shape"]);row["output"]=shape_text(row["output_shape"])
        if name=="stem":
            conv=model.stem
            row["calculation"]=(f"Conv {conv.kernel_size[0]}×{conv.kernel_size[1]} "
                f"{conv.in_channels}→{conv.out_channels}; s={conv.stride[0]},p={conv.padding[0]}; không BN")
        elif name.startswith("block"):
            block=getattr(model,name)
            c=block.expand.in_channels;e=block.expand.out_channels;d=block.se_reduce.out_channels
            h,w=row["input_shape"][2:]
            expected_p=2*c*e+c*(9+25)+9*e*e//8+2*e*d+d+e+2*(c+3*e)
            expected_m=h*w*(2*c*e+c*(9+25)+9*e*e//8)+2*e*d
            canonical=(block.local.kernel_size==(3,3) and block.context.kernel_size==(5,5)
                and block.local.groups==block.context.groups==c
                and block.local.in_channels==block.local.out_channels==c
                and block.context.in_channels==block.context.out_channels==c
                and block.grouped.groups==8 and row["input_shape"]==row["output_shape"]
                and row["parameters"]==expected_p and row["macs"]==expected_m)
            if canonical:
                row["calculation"]=f"B(C={c},E={e},D={d};H={h},W={w}); 4 BN + SiLU + SE + skip"
            else:
                row["calculation"]="Σ Conv/Linear + BN thực tế; xem ledger từng lớp; không áp công thức block gốc"
        elif name.startswith("transition"):
            prefix="t1" if name=="transition1" else "t2"
            conv=getattr(model,prefix+"_conv")
            row["calculation"]=(f"BN({conv.in_channels}) + SiLU + AvgPool2×2,s2 "
                                 f"+ Conv1×1 {conv.in_channels}→{conv.out_channels}")
        elif name=="head":
            row["calculation"]="BN(256) + SiLU + GAP1×1 + flatten + Dropout(p=0); MACs Conv/Linear=0"
        else:
            linear=model.classifier
            row["calculation"]=(f"Linear {linear.in_features}→{linear.out_features}; "
                f"P={linear.in_features}×{linear.out_features}"
                +(f"+{linear.out_features}" if linear.bias is not None else "")
                +f"; MAC={linear.in_features}×{linear.out_features}")
        for metric in cumulative:
            cumulative[metric]+=row[metric]
            row[metric+"_cumulative"]=cumulative[metric]
        rows.append(row)
    for metric in cumulative:
        if cumulative[metric]!=profile[metric]:
            raise ValueError(f"Tổng {metric} theo khối không khớp profiler: {cumulative[metric]} != {profile[metric]}")
    assert len({leaf["name"] for row in rows for leaf in row["leaves"]})==sum(len(row["leaves"]) for row in rows)
    return rows,profile


def export_cumulative_tables():
    import textwrap
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    figures=REPORT_DIR/"figures";figures.mkdir(parents=True,exist_ok=True)
    tables={};markdown=["# Bảng tổng hợp cộng dồn toàn bộ CompactMix-L", "",
        "Input **[1,3,32,32]** (một ảnh CIFAR). Bảng bỏ chiều batch trong hai cột shape. Số liệu lấy từ forward thật; các tổng được đối chiếu trực tiếp với profiler.","",
        "**MACs chỉ đếm Conv/Linear.** BN, activation, pooling, SE nhân và residual cộng có FLOPs bổ sung theo quy ước báo cáo, không gộp vào MAC Conv/Linear. Vì vậy tổng FLOPs lớn hơn 2×MACs. 1 MMac=1.000.000 MAC; 1 GMac=1.000.000.000 MAC.","",
        "## Công thức của B(C,E,D;H,W)","",
        "Với block gốc: E=2C; Conv expand/project 1×1; hai nhánh depthwise 3×3/5×5; grouped Conv3×3 groups=8; SE Conv E→D→E ở 1×1; bốn BN có kênh C,E,E,E.","",
        "```text",
        "P_B = 2*C*E + C*(3*3+5*5) + E*(E/8)*3*3 + 2*E*D + (D+E) + 2*(C+3*E)",
        "      PW đôi    DW đôi       grouped Conv     SE weight  SE bias  BN affine",
        "MAC_B = H*W*[2*C*E + C*(3*3+5*5) + E*(E/8)*3*3] + 2*E*D",
        "         các Conv không gian                              hai Conv SE ở 1x1",
        "```","",
        "Conv chung: P=Cout*(Cin/groups)*kH*kW + Cout nếu có bias; MAC=Hout*Wout*Cout*(Cin/groups)*kH*kW. BN: P=2C, MAC Conv/Linear=0. Linear: P=Fin*Fout+Fout khi có bias, MAC=Fin*Fout.","",
        "**Stem thực tế chỉ có Conv, không có BN.** Bảng này dùng input 32×32 của CompactMix, không dùng input 224×224 hay thông số của hình mẫu.",""]
    for name in DATASETS:
        classes=10 if name=="cifar10" else 100
        actual_model=make_model(classes)
        rows,profile=cumulative_model_accounting(actual_model,classes)
        tables[name]=rows
        frame=pd.DataFrame([{k:v for k,v in row.items() if k not in {"leaves","input_shape","output_shape"}} for row in rows])
        frame.to_csv(REPORT_DIR/f"BANG_CONG_DON_{name.upper()}.csv",index=False,encoding="utf-8-sig")
        leaves=pd.DataFrame([{"block":row["block"],**leaf} for row in rows for leaf in row["leaves"]])
        leaves.to_csv(OUTPUT_ROOT/f"{name}_accounting_leaf_ledger.csv",index=False,encoding="utf-8-sig")
        markdown.extend(["## "+name.upper(),"",
            f"**Tổng: {profile['parameters']:,} Params; {profile['macs']:,} MACs = {profile['macs']/1e9:.9f} GMac; {profile['flops']:,} FLOPs = {profile['flops']/1e9:.9f} GFLOPs.**","",
            "| Khối | Input C×H×W | Output C×H×W | Chi tiết công thức | Params khối | MACs khối | Params cộng dồn | MACs cộng dồn |",
            "|---|---|---|---|---:|---:|---:|---:|"])
        for row in rows:
            markdown.append("| "+" | ".join([row["block"],row["input"],row["output"],row["calculation"],
                f"{row['parameters']:,}",f"{row['macs']:,}",f"{row['parameters_cumulative']:,}",f"{row['macs_cumulative']:,}"])+" |")
        markdown.extend([f"| **TỔNG** | 3×32×32 | {classes} logits | Tổng từng khối, không cộng lại Σ | **{profile['parameters']:,}** | **{profile['macs']:,}** | **{profile['parameters']:,}** | **{profile['macs']:,}** |","",
            "### Kiểm tra công thức tại ba stage","",
            "| Đại diện | C,E,D; H=W | Công thức P đã thế số | P | MACs |",
            "|---|---|---|---:|---:|"])
        for number in (1,3,6):
            row=next(r for r in rows if r["block"]==f"block{number:02}")
            block=getattr(actual_model,f"block{number:02}")
            c,e,d=block.expand.in_channels,block.expand.out_channels,block.se_reduce.out_channels
            h=row["input_shape"][2]
            substituted=(f"2×{c}×{e}+{c}×34+{e}×({e}/8)×9+2×{e}×{d}+({d}+{e})+2×({c}+3×{e})"
                         if row["calculation"].startswith("B(") else "Σ tham số leaf thực tế; block đã chỉnh")
            markdown.append(f"| {row['block']} | {c},{e},{d}; {h} | {substituted} | {row['parameters']:,} | {row['macs']:,} |")
        markdown.extend(["","### Tổng FLOPs theo khối (đối chiếu convention báo cáo)","",
            "| Khối | FLOPs khối | FLOPs cộng dồn |","|---|---:|---:|"])
        markdown.extend(f"| {r['block']} | {r['flops']:,} | {r['flops_cumulative']:,} |" for r in rows)
        markdown.append("")

        # PNG sáu cột giống bảng mẫu; trong hai cột số có thêm dòng Σ cộng dồn.
        data=[]
        for row in rows:
            detail="\n".join(textwrap.wrap(row["calculation"],width=52))
            data.append([row["block"],row["input"],row["output"],detail,
                f"{row['parameters']:,}\nΣ {row['parameters_cumulative']:,}",
                f"{row['macs']/1e6:.6f} M\nΣ {row['macs_cumulative']/1e6:.6f} M"])
        data.append(["TỔNG","3×32×32",str(classes)+" logits","Cộng các khối một lần; không cộng lại cột Σ",
                     f"{profile['parameters']:,}",f"{profile['macs']/1e6:.6f} M"])
        fig,ax=plt.subplots(figsize=(18,13.7));ax.axis("off")
        title=(f"BẢNG TỔNG HỢP CỘNG DỒN COMPACTMIX-L — {name.upper()}\n"
            f"{profile['parameters']:,} Params | {profile['macs']/1e9:.9f} GMac | {profile['flops']/1e9:.9f} GFLOPs | Input 3×32×32")
        ax.set_title(title,fontsize=16,fontweight="bold",pad=30)
        ax.text(.5,.985,"B: PW C→E + (DW3×3 ∥ DW5×5) + GConv3×3(g=8) + SE E→D→E + PW E→C + 4 BN\n"
                "P_B=2CE+34C+9E²/8+2ED+(D+E)+2(C+3E)  |  MAC_B=HW·(2CE+34C+9E²/8)+2ED",
                ha="center",va="center",transform=ax.transAxes,fontsize=10)
        table=ax.table(cellText=data,
            colLabels=["Khối (Block)","Input Shape\nC × H × W","Output Shape\nC × H × W",
                       "Chi tiết công thức / cấu trúc","Số tham số\nParams / Σ Params","Khối lượng tính toán\nMACs / Σ MACs"],
            colWidths=[.085,.095,.095,.405,.145,.175],cellLoc="left",colLoc="center",bbox=[0,.045,1,.91])
        table.auto_set_font_size(False);table.set_fontsize(10)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor("#d4d8df");cell.set_linewidth(.6)
            cell.set_text_props(va="center")
            if r==0:
                cell.set_facecolor("#e5e7eb");cell.set_text_props(weight="bold",ha="center")
            elif r==len(data):
                cell.set_facecolor("#dbeafe");cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#ffffff" if r%2 else "#f7f8fa")
            if c in (1,2,4,5):cell.set_text_props(ha="center")
        ax.text(0,.015,"Σ: tổng cộng dồn đến khối đó. B(C,E,D;H,W): xem công thức trong Markdown/PDF. MACs chỉ đếm Conv/Linear; FLOPs còn gồm BN/activation/pool/MUL/ADD.",
                transform=ax.transAxes,fontsize=9)
        fig.tight_layout()
        png=figures/f"BANG_CONG_DON_{name.upper()}.png"
        fig.savefig(png,dpi=180,bbox_inches="tight")
        plt.show();plt.close(fig)
        display(Markdown("**"+name+" — các giá trị nguyên để kiểm tra tổng**"));display(frame.drop(columns=["calculation"]))
    md_path=REPORT_DIR/"BANG_CONG_DON_THAM_SO_MACS.md"
    md_path.write_text("\n".join(markdown).rstrip()+"\n",encoding="utf-8",newline="\n")
    return tables


BLOCK_ACCOUNTING=export_cumulative_tables()
'''
