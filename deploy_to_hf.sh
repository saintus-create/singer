#!/bin/bash
# 快速部署脚本：将 SoulX-Singer 部署到 Hugging Face Space
# 使用方法: ./deploy_to_hf.sh YOUR_USERNAME YOUR_SPACE_NAME

set -e

if [ $# -lt 2 ]; then
    echo "用法: $0 <YOUR_USERNAME> <YOUR_SPACE_NAME>"
    echo "示例: $0 myusername soulx-singer-demo"
    exit 1
fi

USERNAME=$1
SPACE_NAME=$2
SPACE_REPO="https://huggingface.co/spaces/${USERNAME}/${SPACE_NAME}"

echo "🚀 开始部署到 Hugging Face Space..."
echo "Space: ${USERNAME}/${SPACE_NAME}"
echo ""

# 检查是否已安装 huggingface_hub
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚠️  未检测到 huggingface-cli，正在安装..."
    pip install -U huggingface_hub
fi

# 检查是否已登录
if ! huggingface-cli whoami &> /dev/null; then
    echo "🔐 请先登录 Hugging Face..."
    huggingface-cli login
fi

# 创建 Space（如果不存在）
echo "📦 检查 Space 是否存在..."
if ! huggingface-cli repo info "${USERNAME}/${SPACE_NAME}" --repo-type space &> /dev/null; then
    echo "✨ 创建新的 Space..."
    huggingface-cli repo create "${SPACE_NAME}" --type space --sdk gradio
else
    echo "✅ Space 已存在"
fi

# 检查是否已初始化 Git
if [ ! -d ".git" ]; then
    echo "📝 初始化 Git 仓库..."
    git init
    git add .
    git commit -m "Initial commit for HF Space deployment" || echo "⚠️  没有新文件需要提交"
fi

# 检查远程仓库
if git remote | grep -q "^origin$"; then
    echo "🔄 更新远程仓库地址..."
    git remote set-url origin "${SPACE_REPO}"
else
    echo "➕ 添加远程仓库..."
    git remote add origin "${SPACE_REPO}"
fi

# 推送代码
echo "📤 推送代码到 Hugging Face..."
git push -u origin main || git push -u origin master

echo ""
echo "✅ 部署完成！"
echo "🌐 Space 地址: ${SPACE_REPO}"
echo ""
echo "💡 提示："
echo "   - Space 会自动开始构建，请查看 Logs 标签页"
echo "   - 首次运行会下载模型，可能需要 5-15 分钟"
echo "   - 建议在 Space Settings 中选择 GPU T4 Small 硬件"
