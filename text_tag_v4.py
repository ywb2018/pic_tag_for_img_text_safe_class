import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os
import re  # 用于匹配等级标签
from PIL import Image, ImageTk
import requests
from io import BytesIO

class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("风险标注工具")
        self.root.geometry("1500x1000")
        self.root.minsize(1100, 750)
        
        # 变量初始化
        self.input_folder = ""
        self.output_folder = ""
        self.current_file_index = 0
        self.json_files = []
        self.current_data = None
        self.original_risk_level = ""  # 存储原始风险等级，用于匹配判定
        self.pic_name_to_index = {}    # 图片名 -> 文件索引 映射字典，提速查询
        
        self.create_widgets()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=3)
        main_frame.rowconfigure(2, weight=1)

        # 右上角标注准则
        rule_style = ttk.Style()
        rule_style.configure("Rule.TLabelframe", padding=12)
        rule_style.configure("Rule.TLabelframe.Label", font=("Arial", 14, "bold"))
        rule_frame = ttk.LabelFrame(self.root, text="📌 标注准则", style="Rule.TLabelframe")
        rule_frame.place(relx=0.98, rely=0.02, anchor="ne")
        rule_text = [
            "1. 文本异常(文本与图联动度差/文本不安全) → 直接拒绝",
            "2. 文本无误+等级正确 → ✅等级正确",
            "3. 文本无误+等级错误 → 选L0-L4"
        ]
        for idx, text in enumerate(rule_text):
            lbl = ttk.Label(rule_frame, text=text, font=("Arial",13,"bold"), 
                            foreground="#E53935", anchor="w")
            lbl.pack(anchor="w", pady=3)

        # 顶部控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        folder_frame = ttk.Frame(control_frame)
        folder_frame.pack(fill=tk.X, pady=5)
        ttk.Label(folder_frame, text="输入文件夹:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.input_folder_label = ttk.Label(folder_frame, text="未选择", width=40, relief="sunken", padding=3)
        self.input_folder_label.grid(row=0, column=1, padx=5)
        ttk.Button(folder_frame, text="选择输入文件夹", command=self.select_input_folder).grid(row=0, column=2, padx=5)
        
        ttk.Label(folder_frame, text="输出文件夹:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_folder_label = ttk.Label(folder_frame, text="未选择", width=40, relief="sunken", padding=3)
        self.output_folder_label.grid(row=1, column=1, padx=5)
        ttk.Button(folder_frame, text="选择输出文件夹", command=self.select_output_folder).grid(row=1, column=2, padx=5)
        
        self.load_button = ttk.Button(control_frame, text="加载数据", command=self.load_data, state=tk.DISABLED)
        self.load_button.pack(pady=5)
        
        # 内容显示区（图片+文本）
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, sticky="nsew")
        content_frame.columnconfigure(0, weight=3)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # 图片框
        image_frame = ttk.LabelFrame(content_frame, text="图片预览", padding="10")
        image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        self.image_label = ttk.Label(image_frame, text="图片加载中...", background="#f0f0f0")
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # 右侧文本展示区
        text_frame = ttk.Frame(content_frame)
        text_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        text_frame.rowconfigure(0, weight=2)
        text_frame.rowconfigure(1, weight=2)
        text_frame.rowconfigure(2, weight=6)
        text_frame.columnconfigure(0, weight=1)
        
        # 安全文本展示框
        safe_frame = ttk.LabelFrame(text_frame, text="📄 安全文本", padding="10")
        safe_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        safe_frame.columnconfigure(0, weight=1)
        self.safety_text = tk.Text(safe_frame, height=1, wrap=tk.WORD, font=("Arial",10))
        self.safety_text.grid(row=0, column=0, sticky="nsew")
        s_scroll = ttk.Scrollbar(safe_frame, command=self.safety_text.yview)
        s_scroll.grid(row=0, column=1, sticky="ns")
        self.safety_text.config(yscrollcommand=s_scroll.set)
        
        # 风险等级展示框
        level_frame = ttk.LabelFrame(text_frame, text="⚠️ 风险等级", padding="10")
        level_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        level_frame.columnconfigure(0, weight=1)
        self.level_text = tk.Text(level_frame, height=1, wrap=tk.WORD, font=("Arial",12,"bold"))
        self.level_text.grid(row=0, column=0, sticky="nsew")
        l_scroll = ttk.Scrollbar(level_frame, command=self.level_text.yview)
        l_scroll.grid(row=0, column=1, sticky="ns")
        self.level_text.config(yscrollcommand=l_scroll.set)
        
        # COT内容展示框
        cot_frame = ttk.LabelFrame(text_frame, text="🧠 COT推理内容", padding="10")
        cot_frame.grid(row=2, column=0, sticky="nsew")
        cot_frame.columnconfigure(0, weight=1)
        self.cot_text = tk.Text(cot_frame, wrap=tk.WORD, font=("Arial",10))
        self.cot_text.grid(row=0, column=0, sticky="nsew")
        c_scroll = ttk.Scrollbar(cot_frame, command=self.cot_text.yview)
        c_scroll.grid(row=0, column=1, sticky="ns")
        self.cot_text.config(yscrollcommand=c_scroll.set)
        
        # 底部标注操作区
        bottom_frame = ttk.LabelFrame(main_frame, text="标注操作区", padding="10")
        bottom_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.columnconfigure(2, weight=2)
        bottom_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(bottom_frame)
        left_frame.grid(row=0, column=0, sticky="nw", pady=0)
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(anchor=tk.N, pady=0)
        # 上一张按钮
        self.prev_btn = ttk.Button(nav_frame, text="⬅️ 上一张", command=self.previous_file, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        # 索引跳转组件
        self.index_var = tk.StringVar(value="1")
        self.index_entry = ttk.Entry(nav_frame, textvariable=self.index_var, width=8, font=("Arial",10))
        self.index_entry.pack(side=tk.LEFT, padx=5)
        self.jump_btn = ttk.Button(nav_frame, text="跳转", command=self.jump_to_index, state=tk.DISABLED)
        self.jump_btn.pack(side=tk.LEFT, padx=5)
        
        # 下一张按钮
        self.next_btn = ttk.Button(nav_frame, text="下一张 ➡️", command=self.next_file, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # 图片名查询跳转组件
        self.pic_search_var = tk.StringVar(value="")
        ttk.Label(nav_frame, text="图片名：", font=("Arial",10)).pack(side=tk.LEFT, padx=(15,2))
        self.pic_search_entry = ttk.Entry(nav_frame, textvariable=self.pic_search_var, width=18, font=("Arial",10))
        self.pic_search_entry.pack(side=tk.LEFT, padx=2)
        self.pic_search_btn = ttk.Button(nav_frame, text="查询", command=self.search_by_pic_name, state=tk.DISABLED)
        self.pic_search_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态、进度展示
        self.status_label = ttk.Label(left_frame, text="请选择文件夹加载数据", font=("Arial",9))
        self.status_label.pack(anchor=tk.N, pady=3, fill=tk.X)
        self.annot_status_label = ttk.Label(left_frame, text="标注状态：未标注", foreground="blue", font=("Arial",9,"bold"))
        self.annot_status_label.pack(anchor=tk.N, pady=0, fill=tk.X)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(anchor=tk.N, fill=tk.X, pady=3)

        mid_frame = ttk.Frame(bottom_frame)
        mid_frame.grid(row=0, column=1, sticky="nw", padx=20, pady=0)
        ttk.Label(mid_frame, text="文本判定", font=("Arial",11,"bold")).pack(anchor=tk.N, pady=0)
        self.safety_error_btn = tk.Button(
            mid_frame, text="❌ 文本异常（直接拒绝）",
            bg="#F44336", fg="white", font=("Arial",11,"bold"),
            padx=15, pady=10, relief=tk.RAISED, borderwidth=2,
            command=self.annot_safety_error, state=tk.DISABLED
        )
        self.safety_error_btn.pack(anchor=tk.N, pady=5)

        right_frame = ttk.Frame(bottom_frame)
        right_frame.grid(row=0, column=2, sticky="ne", pady=0)
        ttk.Label(right_frame, text="文本无误 → 等级判定", font=("Arial",11,"bold")).pack(anchor=tk.N, pady=0)
        
        self.level_main_frame = ttk.Frame(right_frame)
        self.level_main_frame.pack(anchor=tk.N, pady=5, fill=tk.X)
        
        self.level_correct_btn = tk.Button(
            self.level_main_frame, text="✅ 等级正确",
            bg="#4CAF50", fg="white", font=("Arial",11,"bold"),
            padx=15, pady=8, relief=tk.RAISED, borderwidth=2,
            command=self.annot_level_correct, state=tk.DISABLED
        )
        self.level_correct_btn.pack(side=tk.LEFT, padx=5)
        
        self.level_btns_frame = ttk.Frame(self.level_main_frame)
        self.level_btns_frame.pack(side=tk.LEFT, padx=5)
        self.level_buttons = {}
        levels = ["L0", "L1", "L2", "L3", "L4"]
        for level in levels:
            btn = tk.Button(
                self.level_btns_frame, text=level, font=("Arial",12,"bold"),
                bg="#2196F3", fg="white", width=5, height=1, padx=3, pady=5,
                command=lambda l=level: self.annot_risk_level(l), state=tk.DISABLED
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.level_buttons[level] = btn
    
    def select_input_folder(self):
        folder = filedialog.askdirectory(title="选择输入文件夹")
        if folder:
            self.input_folder = folder
            self.input_folder_label.config(text=os.path.basename(folder))
            self.check_folder_ready()
    
    def select_output_folder(self):
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.output_folder = folder
            self.output_folder_label.config(text=os.path.basename(folder))
            self.check_folder_ready()
    
    def check_folder_ready(self):
        if self.input_folder and self.output_folder:
            self.load_button.config(state=tk.NORMAL)
    
    def load_data(self):
        if not os.path.exists(self.input_folder):
            messagebox.showerror("错误", "输入文件夹不存在！")
            return
        self.json_files = [f for f in os.listdir(self.input_folder) if f.endswith(".json")]
        if not self.json_files:
            messagebox.showinfo("提示", "无JSON文件！")
            return
        
        # 构建图片名-索引映射字典（提速查询）
        self._build_pic_name_mapping()
        
        os.makedirs(self.output_folder, exist_ok=True)
        self.current_file_index = 0
        self.load_current_file()
        
        # 启用所有功能按钮
        self.safety_error_btn.config(state=tk.NORMAL)
        self.level_correct_btn.config(state=tk.NORMAL)
        for btn in self.level_buttons.values():
            btn.config(state=tk.NORMAL)
        self.prev_btn.config(state=tk.NORMAL if len(self.json_files) >1 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if len(self.json_files) >1 else tk.DISABLED)
        self.jump_btn.config(state=tk.NORMAL)
        self.pic_search_btn.config(state=tk.NORMAL)  # 启用图片查询按钮
    
    def _build_pic_name_mapping(self):
        """遍历所有JSON，提取pic_path中的纯图片名，构建 图片名→文件索引 的映射"""
        self.pic_name_to_index.clear()
        for idx, json_filename in enumerate(self.json_files):
            json_path = os.path.join(self.input_folder, json_filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pic_path = data.get("pic_path", "")
                if pic_path:
                    pure_pic_name = os.path.basename(pic_path)
                    self.pic_name_to_index[pure_pic_name] = idx
            except Exception as e:
                print(f"读取{json_filename}图片路径失败：{str(e)}")
    
    def load_current_file(self):
        if not self.json_files: return
        curr_filename = self.json_files[self.current_file_index]
        input_path = os.path.join(self.input_folder, curr_filename)
        output_path = os.path.join(self.output_folder, curr_filename)
        
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                self.current_data = json.load(f)
            
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    self.current_data = json.load(f)
                self._show_annotated_status()
            else:
                self.annot_status_label.config(text="标注状态：未标注", foreground="blue")
            
            self._update_content_display(curr_filename)
            self.progress_var.set((self.current_file_index+1)/len(self.json_files)*100)
            self.prev_btn.config(state=tk.NORMAL if self.current_file_index>0 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.current_file_index<len(self.json_files)-1 else tk.DISABLED)
            self.index_var.set(str(self.current_file_index + 1))  # 同步索引框
            
        except Exception as e:
            messagebox.showerror("加载失败", f"{curr_filename} 错误：{str(e)}")

    def _extract_core_fields(self, data):
        """提取指定的三个核心字段，兼容两种数据格式"""
        res = {"safe_text": "无安全文本", "risk_level": "未识别等级", "cot": "无COT内容"}
        
        # 1. 提取安全文本
        if "safe_text" in data and data["safe_text"]:
            res["safe_text"] = data["safe_text"]
        elif "filtered_risk_content" in data and len(data["filtered_risk_content"]) > 0:
            res["safe_text"] = data["filtered_risk_content"][0].get("safe_text", "无安全文本")

        # 2. 提取风险等级
        cot_info = data.get("cot_inform", {})
        if cot_info:
            if "risk_level" in cot_info and cot_info["risk_level"]:
                res["risk_level"] = cot_info["risk_level"]
            elif "cot_answer" in cot_info and cot_info["cot_answer"]:
                cot_ans = cot_info["cot_answer"]
                level_match = re.search(r"<answer>(.*?)</answer>", cot_ans)
                if level_match:
                    res["risk_level"] = level_match.group(1).strip()
        self.original_risk_level = res["risk_level"]

        # 3. 提取COT内容
        if "cot_inform" in data and "cot_answer" in data["cot_inform"] and data["cot_inform"]["cot_answer"]:
            res["cot"] = data["cot_inform"]["cot_answer"]

        return res

    def _update_content_display(self, curr_filename):
        if not self.current_data: return
        self.safety_text.delete(1.0, tk.END)
        self.level_text.delete(1.0, tk.END)
        self.cot_text.delete(1.0, tk.END)
        
        core_fields = self._extract_core_fields(self.current_data)
        self.safety_text.insert(1.0, core_fields["safe_text"])
        self.level_text.insert(1.0, core_fields["risk_level"])
        self.cot_text.insert(1.0, core_fields["cot"])
        
        self._display_image()
        self.status_label.config(text=f"{self.current_file_index+1}/{len(self.json_files)} | {curr_filename}")
    
    # 图片加载：等比例铺满
    def _display_image(self):
        self.image_label.config(image="", text="图片加载中...")
        try:
            img_path = self.current_data.get("pic_path", "")
            DEFAULT_IMG_DIR = "F:\\"
            # if not os.path.isfile(img_path):
            #     img_path = os.path.join(DEFAULT_IMG_DIR, img_path)
            if not img_path:
                self.image_label.config(text="无图片路径")
                return
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                self.root.update_idletasks()
                frame_w = self.image_label.winfo_width()
                frame_h = self.image_label.winfo_height()
                if frame_w <=1 or frame_h <=1:
                    frame_w, frame_h = 700, 600
                
                img_ratio = img.width / img.height
                frame_ratio = frame_w / frame_h
                if img_ratio > frame_ratio:
                    new_w = frame_w
                    new_h = int(new_w / img_ratio)
                else:
                    new_h = frame_h
                    new_w = int(new_h * img_ratio)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
            else:
                self.image_label.config(text="图片文件不存在")
            
        except Exception as e:
            self.image_label.config(text=f"图片加载失败：{str(e)[:25]}")
    
    def _show_annotated_status(self):
        """显示标注状态，兼容等级匹配判定"""
        tag = self.current_data.get("tag")
        level = self.current_data.get("manual_risk_level")
        if tag == "fail":
            self.annot_status_label.config(text="标注状态：已拒绝", foreground="red")
        elif level == "correct":
            self.annot_status_label.config(text="✅ 等级正确", foreground="#4CAF50")
        elif level in ["L0","L1","L2","L3","L4"]:
            if level == self.original_risk_level:
                self.annot_status_label.config(text="✅ 等级正确", foreground="#4CAF50")
            else:
                self.annot_status_label.config(text=f"✅ 已校准：{level}", foreground="#2196F3")
        else:
            self.annot_status_label.config(text="标注状态：未标注", foreground="blue")
    
    def annot_safety_error(self):
        self.current_data["tag"] = "fail"
        self.current_data.pop("manual_risk_level", None)
        self._save_annot_result("标注状态：已拒绝", "red")
    
    def annot_level_correct(self):
        self.current_data["manual_risk_level"] = "correct"
        self.current_data.pop("tag", None)
        self._save_annot_result("✅ 等级正确", "#4CAF50")
    
    def annot_risk_level(self, selected_level):
        self.current_data["manual_risk_level"] = selected_level
        self.current_data.pop("tag", None)
        if selected_level == self.original_risk_level:
            self._save_annot_result("✅ 等级正确", "#4CAF50")
        else:
            self._save_annot_result(f"✅ 已校准：{selected_level}", "#2196F3")
    
    def _save_annot_result(self, status_text, color):
        """保存标注结果 + ✅核心恢复：标注完成后自动跳转到下一条"""
        curr_filename = self.json_files[self.current_file_index]
        save_path = os.path.join(self.output_folder, curr_filename)
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.current_data, f, ensure_ascii=False, indent=2)
            self.annot_status_label.config(text=status_text, foreground=color)
            self.root.update()
            
            # ========== 核心恢复：自动跳转下一条逻辑 ==========
            total_count = len(self.json_files)
            # 判断是否为最后一条，非最后一条则自动跳转
            if self.current_file_index < total_count - 1:
                self.current_file_index += 1
                self.load_current_file()
            # 最后一条标注完成，弹窗提示并停止跳转
            else:
                messagebox.showinfo("标注完成", "🎉 所有数据已标注完毕！")
                
        except Exception as e:
            messagebox.showerror("保存失败", f"{str(e)}")
    
    def previous_file(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_current_file()
    
    def next_file(self):
        if self.current_file_index < len(self.json_files)-1:
            self.current_file_index += 1
            self.load_current_file()
        else:
            messagebox.showinfo("完成", "✅ 所有文件已浏览完毕！")
    
    def jump_to_index(self):
        """索引数字跳转功能"""
        input_str = self.index_var.get().strip()
        if not input_str.isdigit():
            messagebox.showwarning("输入错误", "请输入纯数字索引！")
            self.index_var.set(str(self.current_file_index + 1))
            return
        
        target_idx = int(input_str) - 1
        total = len(self.json_files)
        if target_idx < 0 or target_idx >= total:
            messagebox.showwarning("范围错误", f"索引超出范围！请输入 1~{total} 之间的数字")
            self.index_var.set(str(self.current_file_index + 1))
            return
        
        self.current_file_index = target_idx
        self.load_current_file()

    def search_by_pic_name(self):
        """图片名精准查询跳转：输入纯图片名，无需完整路径"""
        input_pic_name = self.pic_search_var.get().strip()
        if not input_pic_name:
            messagebox.showwarning("输入为空", "请输入需要查询的图片名称！")
            self.pic_search_entry.focus()
            return
        
        if input_pic_name in self.pic_name_to_index:
            target_idx = self.pic_name_to_index[input_pic_name]
            self.current_file_index = target_idx
            self.load_current_file()
            self.annot_status_label.config(
                text=f"✅ 匹配成功：{input_pic_name}", 
                foreground="#FF9800"
            )
        else:
            messagebox.showinfo("查询结果", f"未找到图片名为【{input_pic_name}】的数据！")
            self.pic_search_var.set("")
            self.pic_search_entry.focus()

def main():
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()