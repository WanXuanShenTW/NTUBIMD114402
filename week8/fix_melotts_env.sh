#!/bin/bash

echo "🔧 正在修復 NLTK + MeloTTS 所需資源..."

# Step 1: 安裝必要資源
echo "📦 安裝 nltk 資源中..."
python3 -c "
import nltk
for pkg in ['punkt', 'averaged_perceptron_tagger', 'cmudict']:
    try:
        nltk.data.find(f'taggers/{pkg}')
        print(f'✅ 已安裝: {pkg}')
    except LookupError:
        print(f'⬇️  下載: {pkg}')
        nltk.download(pkg)
"

# Step 2: 建立錯誤命名的 tagger_eng 資料夾
NLTK_PATH=$(python3 -c "import nltk; print(nltk.data.find('taggers/averaged_perceptron_tagger').split('/taggers')[0] + '/taggers')")
echo "📁 NLTK 資料路徑為：$NLTK_PATH"

cd "$NLTK_PATH" || { echo "❌ 找不到 NLTK 資料夾！"; exit 1; }

if [ ! -d "averaged_perceptron_tagger_eng" ]; then
    echo "🚧 建立錯誤名稱用資源：averaged_perceptron_tagger_eng"
    mkdir -p averaged_perceptron_tagger_eng
    cp averaged_perceptron_tagger/*.json averaged_perceptron_tagger_eng/

    cd averaged_perceptron_tagger_eng || exit
    for f in averaged_perceptron_tagger.*; do
        mv "$f" "${f/averaged_perceptron_tagger/averaged_perceptron_tagger_eng}"
    done
    echo "✅ 已建立對應資源：averaged_perceptron_tagger_eng"
else
    echo "✅ 資源 already exists：averaged_perceptron_tagger_eng"
fi

echo "🎉 修復完成！你現在可以執行：python TTSme.py"
