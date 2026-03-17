import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # 日本語表示用

# ---------------------------------------------------------
# 1. 設定：ここにExcelファイル名を入れてください
# ---------------------------------------------------------
file_path = 'a20251109ver2processed_data香川.xlsx'   # ← ここを実際のファイル名に変更（例: 'kagawa_rent.xlsx'）
target_col = '物件賃料・価格（円）'       # ← 賃料が入っている列名（適宜変更してください）

# ---------------------------------------------------------
# 2. データの読み込み
# ---------------------------------------------------------
print("データを読み込んでいます...")
try:
    df = pd.read_excel(file_path)
    print(f"読み込み完了: {len(df)}件のデータがあります。")
except FileNotFoundError:
    print("エラー: ファイルが見つかりません。ファイル名を確認してください。")
    exit()

# ---------------------------------------------------------
# 3. 基本統計量の算出（論文の表に使う数値）
# ---------------------------------------------------------
print("\n" + "="*30)
print(f"【{target_col}の分析結果】")
print("="*30)

# 平均値 
mean_val = df[target_col].mean()
# 中央値 
median_val = df[target_col].median()
# 最頻値 
mode_val = df[target_col].mode()[0]
# 最大・最小
max_val = df[target_col].max()
min_val = df[target_col].min()
# 標準偏差 
std_val = df[target_col].std()

print(f"平均値 (Mean)   : {mean_val:,.0f} 円")
print(f"中央値 (Median) : {median_val:,.0f} 円")
print(f"最頻値 (Mode)   : {mode_val:,.0f} 円")
print(f"最小値 - 最大値 : {min_val:,.0f} 円 - {max_val:,.0f} 円")
print(f"標準偏差 (Std)  : {std_val:,.0f}")

# ---------------------------------------------------------
# 4. ヒストグラムの作成と保存（論文の図用）
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# 家賃のヒストグラム（30万円以下に絞って表示すると見やすい）
sns.histplot(df[df[target_col] < 300000][target_col], bins=50, kde=True, color='skyblue')

plt.title(f'香川県 賃貸物件の{target_col}分布', fontsize=16)
plt.xlabel('賃料 (円)', fontsize=14)
plt.ylabel('物件数 (件)', fontsize=14)
plt.grid(axis='y', alpha=0.5)

# グラフを保存
plt.savefig('rent_distribution.png')
print("\nグラフを 'rent_distribution.png' として保存しました。")
plt.show()