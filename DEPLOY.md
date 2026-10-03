# 🚀 部署到 Hugging Face Space 指南

本指南将帮助您将 SoulX-Singer 部署到 Hugging Face Space。

## 📋 前置要求

1. **Hugging Face 账号**：如果没有，请先注册 [huggingface.co](https://huggingface.co/join)
2. **Git**：确保已安装 Git
3. **Hugging Face CLI**（可选但推荐）：`pip install huggingface_hub`

## 🎯 部署步骤

### 方法一：通过 Web 界面创建（推荐）

#### 步骤 1：准备代码仓库

确保您的代码已准备好：
- ✅ `app.py` - Space 入口文件
- ✅ `webui.py` - Gradio 界面代码
- ✅ `requirements.txt` - Python 依赖
- ✅ `README.md` - 包含 Space 配置的 YAML 头部

#### 步骤 2：创建 Space

1. 访问 [huggingface.co/spaces](https://huggingface.co/spaces)
2. 点击 **"Create new Space"** 按钮
3. 填写 Space 信息：
   - **Space name**: 例如 `SoulX-Singer` 或 `soulx-singer-demo`
   - **SDK**: 选择 **Gradio**
   - **Hardware**: 推荐选择 **GPU T4 small**（推理更快，首次下载模型后缓存）
   - **Visibility**: 选择 Public（公开）或 Private（私有）
4. 点击 **"Create Space"**

#### 步骤 3：上传代码

**选项 A：使用 Git 推送（推荐）**

```bash
# 1. 在本地代码目录初始化 Git（如果还没有）
git init
git add .
git commit -m "Initial commit for HF Space"

# 2. 添加 Hugging Face 远程仓库
# 替换 YOUR_USERNAME 和 YOUR_SPACE_NAME
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# 3. 推送代码
git push -u origin main
```

**选项 B：使用 Web 界面上传**

1. 在 Space 页面点击 **"Files and versions"** 标签
2. 点击 **"Add file"** → **"Upload files"**
3. 拖拽或选择以下必需文件：
   - `app.py`
   - `webui.py`
   - `requirements.txt`
   - `README.md`
   - `soulxsinger/` 目录（整个文件夹）
   - `preprocess/` 目录（整个文件夹）
   - `cli/` 目录（整个文件夹）
   - `example/` 目录（整个文件夹）
   - `assets/` 目录（整个文件夹）
   - 其他配置文件（如 `LICENSE`, `.gitignore` 等）

#### 步骤 4：等待构建和首次运行

1. Space 会自动检测到代码并开始构建
2. 查看 **"Logs"** 标签页监控构建进度
3. 首次运行会：
   - 安装 `requirements.txt` 中的依赖
   - 执行 `app.py`
   - **自动下载** `Soul-AILab/SoulX-Singer` 和 `Soul-AILab/SoulX-Singer-Preprocess` 模型（可能需要 5-15 分钟，取决于网络速度）
4. 构建完成后，Space 会自动启动，您可以在 **"App"** 标签页看到界面

### 方法二：使用 Hugging Face CLI

```bash
# 1. 安装 Hugging Face Hub CLI
pip install huggingface_hub

# 2. 登录（会打开浏览器）
huggingface-cli login

# 3. 创建 Space（替换 YOUR_USERNAME 和 YOUR_SPACE_NAME）
huggingface-cli repo create YOUR_SPACE_NAME --type space --sdk gradio

# 4. 克隆 Space 仓库
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# 5. 复制代码文件到 Space 目录
# （将当前代码目录的所有文件复制过来）

# 6. 提交并推送
git add .
git commit -m "Deploy SoulX-Singer to HF Space"
git push
```

## ⚙️ Space 配置说明

Space 配置在 `README.md` 的 YAML 头部：

```yaml
---
title: SoulX-Singer
emoji: 🎤
sdk: gradio
sdk_version: "6.3.0"
app_file: app.py
python_version: "3.10"
suggested_hardware: t4-small  # 取消注释以启用 GPU
---
```

### 硬件选择建议

- **CPU Basic**: 免费，但推理速度较慢，适合测试
- **GPU T4 Small**: 推荐，推理速度快，首次下载模型后缓存
- **GPU T4 Medium/Large**: 适合高并发或更复杂的推理

### 修改硬件配置

1. 进入 Space 页面
2. 点击 **"Settings"** 标签
3. 在 **"Hardware"** 部分选择所需硬件
4. 保存后 Space 会重启

## 🔍 故障排查

### 问题 1：构建失败

**检查点：**
- ✅ `requirements.txt` 中所有依赖版本是否兼容
- ✅ `app.py` 文件是否存在且可执行
- ✅ `README.md` 的 YAML 配置是否正确

**查看日志：**
- 在 Space 页面的 **"Logs"** 标签查看详细错误信息

### 问题 2：模型下载失败

**可能原因：**
- 网络连接问题
- Hugging Face Hub 认证问题

**解决方案：**
- 确保 Space 有网络访问权限（默认有）
- 如果使用私有模型，需要在 Space Settings 中添加 HF Token

### 问题 3：应用启动后无法访问

**检查点：**
- ✅ `app.py` 中 `server_name="0.0.0.0"` 已设置
- ✅ 端口使用环境变量 `PORT`（Space 会自动注入）
- ✅ 查看 **"Logs"** 确认应用是否成功启动

### 问题 4：内存不足

**解决方案：**
- 升级到更大的硬件（T4 Medium/Large）
- 或优化代码，减少内存占用

## 📝 重要提示

1. **首次运行时间**：首次部署时，模型下载可能需要 5-15 分钟，请耐心等待
2. **模型缓存**：下载的模型会缓存在 Space 的存储中，重启后无需重新下载
3. **存储限制**：免费 Space 有存储限制，确保模型文件不会超过限制
4. **自动重启**：Space 会在代码更新后自动重启
5. **日志查看**：遇到问题时，首先查看 **"Logs"** 标签页的详细日志

## 🔗 相关链接

- [Hugging Face Spaces 文档](https://huggingface.co/docs/hub/spaces)
- [Gradio 文档](https://gradio.app/docs/)
- [SoulX-Singer 模型页面](https://huggingface.co/Soul-AILab/SoulX-Singer)
- [SoulX-Singer-Preprocess 模型页面](https://huggingface.co/Soul-AILab/SoulX-Singer-Preprocess)

## ✅ 部署检查清单

部署前确认：
- [ ] `app.py` 文件存在且正确
- [ ] `requirements.txt` 包含所有依赖（包括 `huggingface_hub`）
- [ ] `README.md` 包含正确的 YAML 配置
- [ ] 所有必需的代码文件都已上传
- [ ] `.gitignore` 正确配置（排除 `pretrained_models/` 和 `outputs/`）
- [ ] Space 硬件配置合适（推荐 GPU T4 Small）

部署后验证：
- [ ] Space 构建成功（无错误日志）
- [ ] 模型自动下载完成
- [ ] Web 界面可以正常访问
- [ ] 可以上传音频文件进行测试
- [ ] 推理功能正常工作

---

**祝部署顺利！** 🎉
