"""Inline notebook code that exports an exam-aligned report from verified evidence."""

EXAM_REPORT_CELL = r'''
def export_exam_report():
    from html import escape
    from zipfile import ZipFile
    import xml.etree.ElementTree as ET
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

    exam_path=ROOT/"Final exam.docx"
    if not exam_path.exists(): exam_path=REPORT_DIR/"evidence/Final exam.docx"
    with ZipFile(exam_path) as archive:
        xml=ET.fromstring(archive.read("word/document.xml"))
    ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    original=["".join(t.text or "" for t in p.findall(".//w:t",ns)) for p in xml.findall(".//w:p",ns)]
    original=[line for line in original if line.strip()]
    normalized=" ".join(" ".join(original).split()).casefold()
    for token in ("not over 6m","under 1g","cifar-10","cifar-100","initial learning rates",
                  "sgd","adam","momentum","epochs","pdf","oral exam","same model"):
        if token not in normalized: raise ValueError("Đề đã đổi hoặc thiếu yêu cầu cần kiểm tra: "+token)

    # Đọc lại evidence/hashes lúc xuất; không tin một bảng output cũ trong notebook.
    verified={name:archived_bundle(name) for name in DATASETS}
    comparison=exam_comparison_rows()
    known=[r for r in comparison if r["test_percent"] is not None]
    matrix_complete=all(all(len({r["initial_lr"] for r in known
        if r["dataset"]==name and r["optimizer"]==optimizer})>=2
        for optimizer in ("sgd","adam")) for name in DATASETS)
    checks=[
        ("Thiết kế model L, mô tả kiến trúc","Đã mô tả","CompactMix-L: widths 40/112/256, depths 2/3/8; code Conv/forward tường minh."),
        ("Tham số học được không quá 6M","Đạt","CIFAR-10: 5.248.172; CIFAR-100: 5.271.302; kiểm tra bằng forward/profile."),
        ("FLOPs dưới 1G","Đạt theo convention bài làm","842.406.944 và 842.453.114 FLOPs/ảnh; 1 MAC=2 FLOPs cộng các phép toán bổ sung."),
        ("Train/test CIFAR-10 và CIFAR-100","Đã có ở SGD LR=0,1","Hai selection 45k/5k, 600 epoch; test best trên đủ 10k ảnh mỗi dataset."),
        ("SGD, Adam và các initial LR khác nhau","Đã đủ bảng thực nghiệm" if matrix_complete else "CHƯA HOÀN THÀNH",
         "Xem bảng optimizer × LR × dataset; ô chưa đo giữ trống. Code hỗ trợ chạy không thay thế kết quả thực nghiệm."),
        ("Mô tả learning settings chi tiết","Đã ghi","LR, momentum/betas, decay, epochs, batch, scheduler, loss, augmentation, AMP/EMA, seed và môi trường."),
        ("Nộp báo cáo PDF","Đã xuất PDF","File BAO_CAO_THEO_DE_FINAL_EXAM.pdf; cần điền thông tin sinh viên và kiểm tra trước khi nộp."),
        ("Kiến trúc không trùng nhóm/mạng trước","CẦN ĐỐI CHIẾU","Mô tả cấu trúc riêng; chưa có bằng chứng xác nhận không trùng tất cả nhóm hay mạng đã công bố."),
        ("Vấn đáp và nộp đúng hạn","Sinh viên thực hiện","Vấn đáp 30%; hiệu năng C10 30%, C100 30%; chất lượng PDF 10%. Không qua vấn đáp thì trượt môn."),
    ]
    for name in DATASETS:
        p=PROFILES[name]
        if p["parameters"]>6_000_000 or p["flops"]>=1_000_000_000:
            raise ValueError("Model đang dùng không đạt budget đề: "+name)
        if p["parameters"]!=verified[name]["result"]["parameters"] or p["flops"]!=verified[name]["result"]["flops"]:
            raise ValueError("Model notebook đã đổi so với model của evidence; không gán điểm cũ cho model mới")

    target=REPORT_DIR
    target.mkdir(parents=True,exist_ok=True)
    pdf_path=target/"BAO_CAO_THEO_DE_FINAL_EXAM.pdf"
    md_path=target/"BAO_CAO_THEO_DE_FINAL_EXAM.md"
    csv_path=target/"BANG_SO_SANH_SGD_ADAM_LR.csv"
    audit_path=target/"DOI_CHIEU_FINAL_EXAM.json"
    fonts=Path("C:/Windows/Fonts")
    regular=fonts/"arial.ttf"; bold=fonts/"arialbd.ttf"
    if not regular.exists():
        import matplotlib
        fonts=Path(matplotlib.get_data_path())/"fonts/ttf"
        regular=fonts/"DejaVuSans.ttf"; bold=fonts/"DejaVuSans-Bold.ttf"
    pdfmetrics.registerFont(TTFont("ExamBody",str(regular)))
    pdfmetrics.registerFont(TTFont("ExamBold",str(bold)))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ExamText",fontName="ExamBody",fontSize=9.5,leading=14,spaceAfter=7))
    styles.add(ParagraphStyle(name="ExamTitle",fontName="ExamBold",fontSize=19,leading=25,spaceAfter=16))
    styles.add(ParagraphStyle(name="ExamHeading",fontName="ExamBold",fontSize=13,leading=18,spaceBefore=13,spaceAfter=8))
    styles.add(ParagraphStyle(name="ExamCell",fontName="ExamBody",fontSize=8,leading=11))
    story=[]; markdown=[]

    def paragraph(text):
        story.append(Paragraph(escape(str(text)),styles["ExamText"]))
        markdown.extend([str(text),""])

    def heading(text):
        story.append(Paragraph(escape(text),styles["ExamHeading"]))
        markdown.extend(["## "+text,""])

    def table(headers,rows,widths=None):
        data=[[Paragraph(escape(str(v)),styles["ExamCell"]) for v in headers]]
        data.extend([[Paragraph(escape("—" if v is None else str(v)),styles["ExamCell"]) for v in row] for row in rows])
        tab=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
        tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8eff8")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#c5cdd7")),
            ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.extend([tab,Spacer(1,9)])
        markdown.append("| "+" | ".join(headers)+" |")
        markdown.append("|"+"---|"*len(headers))
        for row in rows:
            markdown.append("| "+" | ".join("—" if v is None else str(v).replace("|","/") for v in row)+" |")
        markdown.append("")

    def image(path,caption,height=240):
        path=Path(path)
        if path.exists():
            from PIL import Image as PILImage
            with PILImage.open(path) as picture:
                w,h=picture.size
            scale=min(475/w,height/h)
            story.append(Image(str(path),width=w*scale,height=h*scale))
            story.append(Paragraph(escape(caption),styles["ExamText"]))
            markdown.extend(["!["+caption+"]("+os.path.relpath(path,target).replace("\\","/")+")",""])

    title="BÁO CÁO FINAL EXAM — COMPACTMIX-L"
    story.append(Paragraph(title,styles["ExamTitle"]))
    markdown.extend(["# "+title,""])
    paragraph("Sinh viên/nhóm: ........................................ | Lớp: ........................ | Giảng viên: ........................")
    paragraph("Nguồn yêu cầu: Final exam.docx. Báo cáo tổ chức theo yêu cầu bắt buộc trong đề; kết quả cũ được kiểm chứng từ evidence, không lấy accuracy demo để thay điểm full-data.")
    paragraph("Trạng thái: còn thiếu thực nghiệm so sánh full-data Adam và các LR khác; cần đối chiếu tính khác biệt kiến trúc. Không tuyên bố đã hoàn thành toàn bộ đề.")
    heading("1. Đối chiếu yêu cầu của đề")
    table(["Yêu cầu","Trạng thái","Bằng chứng / việc còn lại"],checks,[122,100,253])
    paragraph("Đề không quy định số epoch cố định, không bắt buộc 600 epoch, split 45k/5k, retrain 50k, multi-seed, ablation hoặc ngưỡng accuracy 98%/90%. Những mục này là lựa chọn thực nghiệm hoặc mở rộng của dự án. Hiệu năng được chấm theo thứ hạng giữa các nhóm.")

    heading("2. Thiết kế model L và kiến trúc")
    paragraph("CompactMix-L dùng stem Conv3×3 3→40; ba stage 40/112/256 kênh với 2/3/8 MixBlock; transition BN→SiLU→AvgPool2×2→Conv1×1; head BN→SiLU→GlobalAvgPool→Flatten→Linear. Hai dataset dùng chung backbone, head xuất 10 hoặc 100 logits.")
    table(["Khối","Output","Chi tiết"],[
        ["Stem","[N,40,32,32]","Conv3×3, s=1,p=1, không bias"],
        ["Stage 1","[N,40,32,32]","2 MixBlock"],
        ["Transition 1 / Stage 2","[N,112,16,16]","Giảm không gian; 3 MixBlock"],
        ["Transition 2 / Stage 3","[N,256,8,8]","Giảm không gian; 8 MixBlock"],
        ["Head","[N,10] / [N,100]","GAP→flatten→Dropout(p=0)→Linear"]],[130,117,228])
    paragraph("Mỗi MixBlock: BN/SiLU→expand 1×1 C→2C→chia đôi kênh→depthwise 3×3 và 5×5→concat→BN/SiLU→grouped Conv3×3 groups=8→BN/SiLU→SE→projection 1×1 2C→C→DropPath→ADD với skip. SE pool về 1×1, Conv giảm/mở kênh và Sigmoid tạo gate; D=8/14/32 theo stage.")
    image(OUTPUT_ROOT/"architecture.png","Luồng model CompactMix-L",290)
    paragraph("Tính khác biệt: tổ hợp hai kernel trên hai phần kênh, grouped spatial mixing, SE và cấu hình stage được tự triển khai trong bài. Các thành phần Conv/SE/residual đã có trong tài liệu. Cần trình bày nguồn tham khảo và đối chiếu kiến trúc với nhóm khác/mạng liên quan; chưa có cơ sở khẳng định tính mới hay độc nhất.")

    heading("3. Kiểm chứng tham số và FLOPs")
    profile_rows=[]
    for name in DATASETS:
        p=PROFILES[name]
        profile_rows.append([name,f"{p['parameters']:,}",f"{p['macs']:,}",f"{p['flops']:,}",f"{p['flops']/1e9:.6f}"])
    table(["Dataset","Parameters","MACs/ảnh","FLOPs/ảnh","GFLOPs"],profile_rows,[67,101,101,111,95])
    paragraph("Conv: P=Cout×(Cin/groups)×kH×kW, thêm Cout nếu có bias. Linear: P=Fin×Fout+Fout nếu có bias. BN affine: P=2C. MAC Conv=N×Hout×Wout×Cout×(Cin/groups)×kH×kW. Profiler dùng N=1, 1 MAC=2 FLOPs, cộng bias/BN/SiLU/Sigmoid/pooling/SE MUL/residual ADD; bỏ qua memory operations split/cat/reshape. Đây là ước lượng inference, không phải latency hoặc training FLOPs.")
    paragraph("Đếm trực tiếp model.parameters(), thực chạy forward để lấy shape và kiểm tra backward. Cả hai head đạt ≤6M tham số và <1G FLOPs theo convention trên. Code và bảng layer đầy đủ nằm trong notebook/CSV của output.")

    heading("4. Dữ liệu và learning settings")
    paragraph("CIFAR-10/100 có 50.000 ảnh train và 10.000 test, RGB 32×32. Bài chọn stratified split 45.000 train / 5.000 validation với split seed 2026; seed khởi tạo 42. Không giao nhau, phủ đủ 50k indices. Validation dùng chọn best; hai checkpoint best 45k được đánh giá trên đủ 10k test theo yêu cầu người dùng trước đó.")
    paragraph("Train: RandomCrop(32,padding=4,reflect), HorizontalFlip, AutoAugment CIFAR10 policy cho cả hai dataset, ToTensor, Normalize theo mean/std riêng. Validation/test chỉ ToTensor/Normalize; không TTA, ensemble, pretrained hoặc dữ liệu ngoài trong hai phép đo đã có.")
    settings_rows=[
        ["Optimizer / initial LR","SGD / 0,1"],["Momentum / Nesterov","0,9 / True"],
        ["Weight decay","0,0005 cho tham số ndim>1; 0 cho nhóm còn lại"],
        ["Epochs / batch size","600 / 128"],["Scheduler / warmup","Cosine / 5 epoch"],
        ["Loss","Soft-label cross entropy; label smoothing=0,05"],
        ["Mixup / CutMix / xác suất","0,2 / 1,0 / 0,75; tắt mixing ở đoạn cuối lịch"],
        ["EMA decay / clipping","Tối đa 0,999 / grad norm 5"],["AMP / deterministic","CUDA AMP / True"],
        ["DropPath / head dropout","Tăng dần 0→0,1 / 0"],["Workers đã train / notebook","2 / 0 (Jupyter Windows)"],
        ["Seed model / split","42 / 2026"],
        ["Adam được hỗ trợ","betas=(0,9;0,999), eps=1e-8; LR kế hoạch 0,0003/0,001/0,003; chưa có full-data result"],
    ]
    table(["Learning setting","Giá trị của thí nghiệm/bản kế hoạch"],settings_rows,[150,325])
    for name in DATASETS:
        cfg=verified[name]["config"]
        if cfg["optimizer"]!="sgd" or cfg["lr"]!=.1 or cfg["epochs"]!=600 or cfg["batch_size"]!=128:
            raise ValueError("Learning settings archive đã đổi; cần cập nhật nội dung báo cáo")
    paragraph("Train loss dùng soft targets; mixed accuracy là tỷ lệ đúng có trọng số theo nhãn trộn. Validation/test dùng nhãn cứng và cross entropy chuẩn. Không coi train mixed accuracy là accuracy sạch, không diễn giải hai loss là hoàn toàn cùng bài toán.")
    env=verified[DATASETS[0]]["result"]["metadata"]["environment"]
    paragraph("Môi trường thí nghiệm đã lưu: "+json.dumps(env,ensure_ascii=False)+". Notebook dùng chính Microsoft Store Python 3.13 hiện tại; không tạo môi trường khác. Config/runtime/source/split/seed được lưu cùng checkpoint.")

    heading("5. Kết quả CIFAR-10 và CIFAR-100 đã đo")
    result_rows=[]
    for name in DATASETS:
        b=verified[name]; t=b["test"]; m=t["test_metrics"]
        result_rows.append([name,f"{t['epoch']} / {t['weight_kind']}",t["validation_accuracy"],m["accuracy"],m["top5_accuracy"],f"{m['macro_f1']:.6f}",f"{t['correct']}/{t['samples']}"])
    table(["Dataset","Best epoch","Val %","Test %","Top-5 %","Macro-F1","Đúng/tổng"],result_rows,[60,76,50,50,60,77,102])
    paragraph("Phạm vi: hai điểm test là của best selection train 45k, không phải demo subset và không phải retrain 50k. Điểm được kiểm chứng bằng SHA256 checkpoint, số ảnh và đường chéo confusion matrix. Không tự đo thêm test trong bước xuất báo cáo.")
    for name in DATASETS:
        b=verified[name]; r=b["result"]; h=b["history"]
        paragraph(f"{name}: train/validation đã hoàn thành {r['epochs']} epoch; tổng thời gian engine {r['training_time']:.2f} giây, trung bình {r['training_time']/r['epochs']:.2f} giây/epoch; tổng AMP skipped updates {int(h.amp_skips.sum())}. Test loss={b['test']['test_metrics']['loss']:.6f}.")
        image(OUTPUT_ROOT/f"{name}_archived_curves.png",name+" — lịch sử đầy đủ 600 epoch",185)
        image(OUTPUT_ROOT/f"{name}_archived_confusion_recall.png",name+" — confusion matrix và recall thấp nhất",200)
    paragraph("Accuracy=100×correct/total; precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2PR/(P+R); macro-F1 là trung bình các lớp. Confusion dùng hàng=true, cột=predicted. Precision/recall/F1/support toàn bộ lớp có trong CSV; top-5 là nhãn thật thuộc 5 logits cao nhất.")

    heading("6. So sánh SGD / Adam và các initial learning rates")
    paragraph("Đây là yêu cầu thực nghiệm trong đề, hiện chưa đủ bằng chứng hoàn tất. Các LR cụ thể là lựa chọn của bài; đề nêu 0,1 và 0,15 làm ví dụ, không ấn định số run hoặc số epoch. Bảng dưới có kế hoạch 3 LR mỗi optimizer trên 2 dataset; không xem những ô trống là accuracy bằng 0.")
    table(["Dataset","Opt","LR","Epochs","Best val %","Test %","Trạng thái"],
        [[r["dataset"],r["optimizer"],r["initial_lr"],r["epochs"],r["validation_percent"],r["test_percent"],r["status"]] for r in comparison],
        [60,45,48,46,67,59,150])
    paragraph("Giữ cùng kiến trúc, split, seed và ngân sách epoch khi so sánh; ghi đầy đủ settings mỗi run. Kế hoạch hiện dùng weight decay SGD=0,0005 và Adam=0,0001; nếu giữ khác nhau thì kết luận là so sánh recipe optimizer, không cô lập riêng tác động optimizer. Chọn candidate bằng validation, không dùng test để tối ưu LR/kiến trúc. Không có đủ kết quả để kết luận SGD tốt hơn Adam hoặc LR nào tốt nhất.")

    heading("7. Kết luận theo đúng mức bằng chứng")
    paragraph("CompactMix-L đạt ngân sách ≤6M/<1G theo profiler và đã train/test cả CIFAR-10, CIFAR-100 với SGD LR=0,1. Test best train 45k đạt 97,18% và 82,98%. Phần thực nghiệm so sánh Adam/nhiều LR chưa hoàn thành; cần đối chiếu tính khác biệt kiến trúc. Vì vậy báo cáo chưa tuyên bố hoàn thành toàn bộ yêu cầu final exam.")
    paragraph("Retrain 50k, multi-seed và ablation là mở rộng, không phải thiếu sót bắt buộc theo văn bản đề. Quy tắc giữ kín test và freeze cho thí nghiệm mới vẫn tuân theo protocol repository. Các phép đo latency, macro-F1 hoặc sơ đồ chi tiết hỗ trợ giải thích nhưng không thay thế so sánh optimizer/LR.")

    heading("8. Chuẩn bị vấn đáp và tài liệu kiểm chứng")
    paragraph("Cần giải thích được: công thức shape Conv và tham số; groups/depthwise; concat khác ADD; SE multiply/broadcast; BN train/eval; SiLU; loss/optimizer; momentum; LR/warmup/cosine; EMA; best/last và resume; validation khác test; phạm vi của điểm số đã báo cáo. Theo đề, không qua vấn đáp thì trượt môn dù hiệu năng tốt.")
    table(["Dataset","SHA256 best đã đo test"],[[name,verified[name]["test"]["checkpoint_sha256"]] for name in DATASETS],[72,403])
    paragraph("Code đầy đủ: CompactMix_CIFAR10_CIFAR100_BAO_CAO.ipynb. Evidence: report/evidence/cifar10 và cifar100; checkpoint best: report/checkpoints; bảng optimizer/LR: BANG_SO_SANH_SGD_ADAM_LR.csv. PDF snapshot trước: Bao_cao_CIFAR10_CIFAR100.pdf được giữ nguyên.")
    paragraph("Tham khảo: PyTorch Training a Classifier (mẫu bố cục notebook); PyTorch nn API; residual learning (He et al., 2015); squeeze-and-excitation (Hu et al., 2017); grouped/depthwise convolution trong tài liệu liên quan. Các khối chuẩn không tự chứng minh kiến trúc độc nhất; cần đối chiếu thiết kế cuối với mạng tham khảo và các nhóm.")

    heading("Phụ lục A. Nguyên văn yêu cầu từ Final exam.docx")
    for line in original:paragraph(line)
    exam_sha=hashlib.sha256(exam_path.read_bytes()).hexdigest()
    paragraph("SHA256 đề bài: "+exam_sha)

    def footer(canvas,document):
        canvas.setFont("ExamBody",8)
        canvas.drawString(40,25,"CompactMix-L · Final exam · kết quả đã kiểm chứng, phần thiếu ghi rõ")
        canvas.drawRightString(A4[0]-40,25,str(document.page))
    document=SimpleDocTemplate(str(pdf_path),pagesize=A4,rightMargin=40,leftMargin=40,
        topMargin=38,bottomMargin=40,title=title,author="Sinh viên / nhóm")
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    md_path.write_text("\n".join(markdown).rstrip()+"\n",encoding="utf-8")
    pd.DataFrame(comparison).to_csv(csv_path,index=False,encoding="utf-8-sig")
    audit={"exam_docx_sha256":exam_sha,"source":"Final exam.docx","requirements":[
        {"requirement":a,"status":b,"evidence":c} for a,b,c in checks],
        "explicit_exam_requirements":["<=6M learnable parameters","<1G FLOPs","CIFAR10 and CIFAR100",
            "SGD and Adam with different initial LRs","detailed learning settings","PDF report",
            "architecture distinction","oral exam and submission"],
        "project_extensions_not_explicit_exam_requirements":["600 epochs","45k/5k split","50k retraining",
            "multiple seeds","ablations","98%/90% accuracy targets"],
        "comparison":comparison,"archived_checkpoint_sha256":{n:b["test"]["checkpoint_sha256"] for n,b in verified.items()},
        "pdf_sha256":hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "complete_optimizer_lr_evidence":matrix_complete}
    atomic_json(audit_path,audit)
    return {"pdf":str(pdf_path),"markdown":str(md_path),"comparison_csv":str(csv_path),
            "audit":str(audit_path),"complete_optimizer_lr_evidence":matrix_complete}

EXAM_REPORT_FILES=export_exam_report()
display(pd.DataFrame([EXAM_REPORT_FILES]))
'''
