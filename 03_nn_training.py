import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import seaborn as sns
from datetime import datetime
import sys
import atexit

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'MS Gothic'


# ==========================================
# ▼ 設定スイッチ（ここだけ変える）
# ==========================================
# ★ここを既存の名前にすれば「ロード（続き）」
# ★ここを新しい名前に変えれば「初期化（新規）」になります
model_filename = "kyouto_model.pth"

# 学習または追加学習

# --- 設定部分 ---
add_epochs = 1000



# ------------------------------------------
# ▼ 以下は自動設定エリア（触らなくてOK）
# ------------------------------------------
# 保存用ディレクトリ作成
output_dir = "neural_network_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# パスを確定
model_path = os.path.join(output_dir, model_filename)


# 保存用ディレクトリ作成（絶対パスに変更
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
runoutput_dir = os.path.join(os.getcwd(), f"neural_network_outputs_{timestamp}")
if not os.path.exists(runoutput_dir):
    os.makedirs(runoutput_dir)
    print(f"出力フォルダを作成しました: {runoutput_dir}")

# すべての標準出力/標準エラーをファイルにも保存するシンプルなTee
log_dir = runoutput_dir  # モデル保存先と同じ場所にログも保存
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + "_run.txt")

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

log_file = open(log_path, "a", encoding="utf-8")
orig_stdout, orig_stderr = sys.stdout, sys.stderr
sys.stdout = Tee(orig_stdout, log_file)
sys.stderr = Tee(orig_stderr, log_file)

prev_excepthook = sys.excepthook

def _cleanup_on_error(exc_type, exc, tb):
    # エラー時はログを残さず削除する
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    try:
        log_file.close()
        if os.path.exists(log_path):
            os.remove(log_path)
    finally:
        prev_excepthook(exc_type, exc, tb)

sys.excepthook = _cleanup_on_error
atexit.register(log_file.close)
print(f"ログファイル: {log_path}")

# ==========================================
# 1. データ読み込みと前処理
# ==========================================
excel_file = "データ変更5_2022京都賃貸データ.xlsx"
data = pd.read_excel(excel_file)
print(excel_file, "を読み込みました.")


# 特徴量とターゲットの列名を指定
feature_columns = ["履歴ID", "築年", "専有・建物面積（平米）", "設備数","緯度","経度","管理費（円）",
                   "共益費（円）","徒歩(分)","バス(分)", "建物構造_ブロック造","建物構造_木造","建物構造_他","建物構造_ＡＬＣ","建物構造_ＰＣ","建物構造_ＲＣ","建物構造_ＳＲＣ","建物構造_軽量鉄骨造","建物構造_鉄骨造","建物構造_ＨＰＣ",
                   "所在地名1_与謝郡与謝野町","所在地名1_久世郡久御山町","所在地名1_乙訓郡大山崎町","所在地名1_亀岡市","所在地名1_京丹後市","所在地名1_京田辺市","所在地名1_京都市上京区",
                   "所在地名1_京都市下京区","所在地名1_京都市中京区","所在地名1_京都市伏見区","所在地名1_京都市北区","所在地名1_京都市南区","所在地名1_京都市右京区","所在地名1_京都市山科区",
                   "所在地名1_京都市左京区","所在地名1_京都市東山区","所在地名1_京都市西京区","所在地名1_八幡市","所在地名1_南丹市","所在地名1_向日市","所在地名1_宇治市","所在地名1_宮津市",
                   "所在地名1_木津川市","所在地名1_相楽郡精華町","所在地名1_福知山市","所在地名1_綴喜郡井手町","所在地名1_綴喜郡宇治田原町","所在地名1_綾部市","所在地名1_舞鶴市","所在地名1_船井郡京丹波町","所在地名1_長岡京市",
                   "間取りタイプ_ターゲットエンコーディング_中央値"]

target_column = "物件賃料・価格（円）"


print(feature_columns)

# シード値を固定
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# dataのインデックスを保存
data = data.reset_index(drop=True)
indices = data.index.values

# 特徴量とターゲットに分割
X = data[feature_columns].values
y = data[target_column].values



# データチェック
print("X finite:", np.isfinite(X).all())
print("y finite:", np.isfinite(y).all())
stds = X.std(axis=0)
zero_var_cols = [feature_columns[i] for i, s in enumerate(stds) if s == 0]
print("Zero variance columns:", zero_var_cols)
bad_mask = ~np.isfinite(X)
bad_cols = [feature_columns[i] for i in np.where(bad_mask.any(axis=0))[0]]
bad_rows = np.where(bad_mask.any(axis=1))[0]
print("Bad columns:", bad_cols[:20])
print("Bad rows count:", len(bad_rows))
print("Sample bad rows:", bad_rows[:5])
if len(bad_rows) > 0:
    print(data.loc[bad_rows[:5], bad_cols].head())
    
    
# データの標準化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.1, random_state=42
)



# ==========================================
# テンソルに変換（ここで float32 に変換されます）
# ==========================================
# 【重要】
# Pandas(エクセル)は精密な「float64(倍精度)」でデータを扱いますが、
# PyTorchの学習は通常、計算速度とメモリ節約のために「float32(単精度)」で行います。
# 
# 下のコードの `dtype=torch.float32` により、強制的に桁数が削られるため、
# 逆変換して確認した時に「43.7 vs 43.6」のような微細な表示ズレが発生します。
# これは故障ではなく、この変換による正常な仕様です（学習には影響ありません）。

# テンソルに変換
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# ==========================================
# 2. モデル定義と学習ループ
# ==========================================
class HousePriceNN_Overfit(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 10000, bias=True),
            nn.ReLU(),
            nn.Linear(10000, 1, bias=True),
        )

    def forward(self, x):
        return self.model(x)

model = HousePriceNN_Overfit(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)



# =================================================================
# 【メモ】保存先をフォルダで整理するコード（必要になったら使う）
# =================================================================
#
# # 1. 保存先のフォルダ名を決める
# # メリット：ファイルをむき出しで置くと散らかるので、専用の場所を作って整理整頓する
# output_dir = "neural_network_outputs"
#
# # 2. そのフォルダが存在するか確認し、なければ作成する
# # メリット：これがないと、初回実行時に「保存先のフォルダがない」と怒られてエラーになるのを防ぐ
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # 3. フォルダ名とファイル名を合体させて、正式なパスを作る
# # メリット：os.path.join を使うと、Windows(\)やMac(/)の違いを気にせず正しいパスを作ってくれる
# # 保存場所イメージ： neural_network_outputs\あ20251110kaghawa_特徴量変化house_price_model.pth
# model_path = os.path.join(output_dir, "あ20251110kaghawa_特徴量変化house_price_model.pth")




# ==========================================
# ▼ 学習モデルの保存・読み込み設定 ▼
# ==========================================
# ★重要★
# ここで指定したファイル名が「フォルダ内に存在するかどうか」で、
# この後のプログラムの動き（続きから or 最初から）が決まります。
#
# 1. 【続きから学習したい場合】（ロード）
#    前回保存したファイル名をそのまま指定してください。
#    → 「ファイルあり」と判定され、データを読み込んで続きからスタートします。
#
# 2. 【最初から学習したい場合】（新規・初期化）
#    まだ存在しない、新しい名前に書き換えてください。（例: "...model_v2.pth" など）
#    → 「ファイルなし」と判定され、初期化して0からスタートします。
#







# 今回追加で行う学習回数、これは初期化時も同じである。

# 読み込み処理
if os.path.exists(model_path):
    print(f"保存済みモデル({model_path})が見つかりました。読み込みます。")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 続きからなので、過去の記録を引き継ぐ
    start_epoch = checkpoint.get('epoch', 0)
    losses = checkpoint.get('losses', [])
    print(f"前回 {start_epoch} エポックまで学習済み。続きから {add_epochs} エポック学習します。")
else:
    print("保存済みモデルが見つかりません。新規学習を開始します。")
    # 新規なので初期化
    start_epoch = 0
    losses = []

# --- 学習ループ（共通化） ---
model.train()
target_epoch = start_epoch + add_epochs # 最終的に何エポックまでやるか

for epoch in range(start_epoch, target_epoch):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # 表示（現在のエポック数を表示）
    current_epoch = epoch + 1
    if current_epoch % 10 == 0:
        print(f"Epoch {current_epoch}/{target_epoch}, Loss: {loss.item():.4f}")

# --- 保存処理（共通化） ---
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': target_epoch, # 今回終わった時点のエポック数
    'losses': losses
}, model_path) # 必ず model_path に保存して、次回読み込めるようにする

print("モデルと学習履歴を保存しました。")

# 2. 日付付きの名前でも「コピー」として保存（バックアップ用）
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_filenameから拡張子を除去
model_name_base = os.path.splitext(model_filename)[0]
# エポック数も含めてより詳細に
backup_filename = f"{time_str}_{model_name_base}_epoch{target_epoch}_backup.pth"
backup_path = os.path.join(output_dir, backup_filename)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': target_epoch,
    'losses': losses
}, backup_path)
print(f"バックアップを保存しました: {backup_filename}")

# os.path.splitext は、ファイル名の「一番最後のドット」を見つけて分割する仕事人です
# 例: "latest_model.pth" → ("latest_model", ".pth") に分割され、
# [0] を指定することで、拡張子を除いた "latest_model" だけを取得しています


# ==========================================
# 3. 評価とデータ集計 (ループを統合)
# ==========================================


# ==============================================================================
# 【用語解説】 describe() の出力結果（パーセンテージ）の見方
# ==============================================================================
# この表は、データを「誤差が小さい順（優秀な順）」に並べたときの統計情報です。
#
# ■ count : データの件数
# ■ mean  : 平均値（全ての誤差を足して個数で割った値）
# ■ std   : 標準偏差（誤差のバラつき。値が大きいほど予測が不安定）誤差の平均からのズレを表している。
# ■ min   : 最小値（ベストスコア。一番うまく予測できたデータの誤差）
#
# --- 以下の％は「順位」を表します（金額の割引率ではありません） ---
#
# ■ 25% (第1四分位数)
#   意味：「成績優秀な上位25%のデータは、誤差がこの金額以内に収まっている」
#   → ここが小さいほど、精度の高い予測が多いことを示します。
#
# ■ 50% (中央値 / メジアン) ★最重要
#   意味：「ちょうど真ん中の順位のデータ。半分のデータは誤差がこの金額以内」
#   → 「平均値(mean)」は大外れデータに引っ張られやすいため、
#      モデルの本来の実力を知るには、この「50%」を見るのが一番確実です。
#
# ■ 75% (第3四分位数)
#   意味：「全体の75%（大部分）のデータは、誤差がこの金額以内に収まっている」
#   → 逆に言うと、残りの25%はこれ以上に大きく外している（苦手なデータ）ということです。
#
# ■ max   : 最大値（ワーストスコア。一番大きく外してしまったデータの誤差）
# ==============================================================================



# -------------------------------------------------------
# 1. モード切替: 「学習モード」から「テストモード」へ
# -------------------------------------------------------
# model.eval()
#   - 役割: DropoutやBatch Normalizationなどの「学習時専用の挙動」を停止します。
#   - 理由: これをしないと、テスト中なのに学習時の振る舞いをしてしまい、正しい実力が測れません。

model.eval()
results_data = []

with torch.no_grad():
    # まとめて推論（ループより高速）
    predictions = model(X_train)
    
    # スケールを戻す
    original_targets = scaler_y.inverse_transform(y_train.numpy())
    original_predictions = scaler_y.inverse_transform(predictions.numpy())
    
    for i in range(len(X_train)):
        actual_val = original_targets[i][0]
        pred_val = original_predictions[i][0]
        diff = abs(actual_val - pred_val)
        data_idx = idx_train[i] # 元データのインデックス
        
        # 基本情報
        row = {
            "Index": data_idx,
            "履歴ID": data.loc[data_idx, "履歴ID"],
            "Actual Target": actual_val,
            "Predicted Value": pred_val,
            "Difference": diff
        }
        
        # 分析用の特徴量を追加 (スケーリング済みの値ですが分布確認には使えます)
        # 必要であればここで inverse_transform を X についても行う検討が必要ですが
        # 今回は「傾向を見る」ためにそのまま、あるいは元のdataから取得します
        
        # 元のdataフレームから生の値を取得して格納（こちらの方が確実です）
        for col in ["築年", "専有・建物面積（平米）", "設備数", "徒歩(分)"]:
            row[col] = data.loc[data_idx, col]
            
        results_data.append(row)

# データフレーム化
results_df = pd.DataFrame(results_data)

# ==============================================================================
# 【重要】 Pandasの魔法のコマンド .describe()
# ==============================================================================
# 以下の1行を実行するだけで、ライブラリ(Pandas)が裏側で全データを走査し、
# 複雑な統計計算（平均、分散、標準偏差、四分位数など）を全て自動でやってくれます。
#
# もし手動でやるなら「(x - mean) ** 2 ...」のような長い数式を書く必要がありますが、
# ここでは「std (標準偏差)」として自動計算された結果が表示されます。
# ==============================================================================

# 結果表示
print("\n=== 予測結果の概要 ===")
pd.set_option('display.float_format', '{:.2f}'.format)
print(results_df[["Actual Target", "Predicted Value", "Difference"]].describe())

# ==========================================
# 4. エクセル出力 (差分が大きいデータ)
# ==========================================
diff_threshold = 3000
diff_over_threshold = results_df[results_df["Difference"] >= diff_threshold].copy()

if not diff_over_threshold.empty:
    # 履歴IDを使って元のdataの全列と結合
    excel_output = pd.merge(
        diff_over_threshold,
        data,
        on="履歴ID",
        how="left",
        suffixes=('', '_original') # 重複列名の処理
    )
    
    # 出力したい列を整理（重複を避けるため columns をフィルタリングしても良いですが、一旦全保存）
    save_path = "diff_over_3000ver0805.xlsx"
    excel_output.to_excel(save_path, index=False)
    print(f"\n差分が{diff_threshold}円以上のデータを '{save_path}' に保存しました。（件数: {len(excel_output)}）")
else:
    print(f"\n差分が{diff_threshold}円以上のデータはありませんでした。")




    
# ==========================================
# ★追加機能：訓練データ vs テストデータ 比較評価
# ==========================================
print("\n=== 過学習（Overfitting）の確認 ===")

model.eval() # 評価モード

with torch.no_grad():
    # 1. 訓練データでの予測（練習問題の成績）
    pred_train = model(X_train)
    loss_train = criterion(pred_train, y_train).item()
    
    # 2. テストデータでの予測（初見問題の成績）★ここが重要
    pred_test = model(X_test)
    loss_test = criterion(pred_test, y_test).item()

    # --- スケールを実際の金額（円）に戻す ---
    # 訓練
    actual_train_yen = scaler_y.inverse_transform(y_train.numpy())
    pred_train_yen = scaler_y.inverse_transform(pred_train.numpy())
    diff_train = np.abs(actual_train_yen - pred_train_yen)
    
    # テスト
    actual_test_yen = scaler_y.inverse_transform(y_test.numpy())
    pred_test_yen = scaler_y.inverse_transform(pred_test.numpy())
    diff_test = np.abs(actual_test_yen - pred_test_yen)

# --- 成績表の表示 ---
print(f"{'Data Type':<10} | {'MSE Loss':<10} | {'平均誤差(MAE)':<15} | {'最大誤差':<15}")
print("-" * 60)
print(f"{'Train(練習)':<10} | {loss_train:.4f}     | {np.mean(diff_train):,.0f}円          | {np.max(diff_train):,.0f}円")
print(f"{'Test (本番)':<10} | {loss_test:.4f}     | {np.mean(diff_test):,.0f}円          | {np.max(diff_test):,.0f}円")
print("-" * 60)

# --- 判定ロジック ---
mae_gap = np.mean(diff_test) - np.mean(diff_train)
if mae_gap > 30000: # 基準は要調整（例: 平均誤差の差が3万円以上なら過学習疑い）
    print("【判定】過学習の可能性が高いです (Testの誤差が Trainより大幅に大きい)")
elif mae_gap < 0:
    print("【判定】Testの方が成績が良いです（データ数が少ない場合などの珍しいケース）")
else:
    print("【判定】汎化性能は悪くありません（誤差の差が許容範囲内）")


# --- 比較グラフの作成 ---
plt.figure(figsize=(10, 8))

# 訓練データ（青色・薄く）
plt.scatter(actual_train_yen, pred_train_yen, color='blue', alpha=0.1, label='Train Data (Learned)', s=10)

# テストデータ（赤色・濃く）
plt.scatter(actual_test_yen, pred_test_yen, color='red', alpha=0.7, label='Test Data (Unseen)', marker='x', s=30)

# 理想線 (y=x)
min_val = min(actual_train_yen.min(), actual_test_yen.min())
max_val = max(actual_train_yen.max(), actual_test_yen.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
plt.xlim(0, 120000)
plt.ylim(0, 120000)
plt.xlabel("Actual Price (Yen)")
plt.ylabel("Predicted Price (Yen)")
plt.title("過学習チェック: 訓練データ vs テストデータ")
plt.legend()
plt.tight_layout()

# 保存
save_path_overfit = os.path.join(runoutput_dir, "04_overfitting_check.png")
plt.savefig(save_path_overfit, dpi=150, bbox_inches='tight')
plt.close()
print(f"比較グラフを保存しました: {save_path_overfit}")

# ==========================================
# 5. グラフ描画と保存 (統合処理)
# ==========================================
print("\n=== グラフの生成と保存を開始します ===")

# 配列の準備（Seaborn用）
actual_arr = np.array(results_df["Actual Target"])
pred_arr = np.array(results_df["Predicted Value"])

# --- Graph 1: Lossの推移 ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses, label="Loss")
min_val = min(actual_arr.min(), pred_arr.min())
max_val = max(actual_arr.max(), pred_arr.max())
plt.xlim(0, 120000)
plt.ylim(0, 120000)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Lossの推移")
plt.legend()
plt.grid()
plt.tight_layout()
save_path = os.path.join(runoutput_dir, "01_loss_history.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"保存完了: {save_path}")

# --- Graph 2: 散布図 (全体) ---
plt.figure(figsize=(10, 8))
plt.scatter(actual_arr, pred_arr, color='blue', alpha=0.5, label="Data Points")
min_val = min(actual_arr.min(), pred_arr.min())
max_val = max(actual_arr.max(), pred_arr.max())
plt.xlim(0, 120000)
plt.ylim(0, 120000)
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y=x)")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("予測値 vs 実際値 (散布図)")
plt.legend()
plt.tight_layout()
save_path = os.path.join(runoutput_dir, "02_scatter_plot.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"保存完了: {save_path}")

# --- Graph 3: Seaborn Histplot (修正版ヒートマップ) ---
plt.figure(figsize=(10, 8))
sns.histplot(x=actual_arr, y=pred_arr, bins=50, cmap='Reds', cbar=True)
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
min_val = min(actual_arr.min(), pred_arr.min())
max_val = max(actual_arr.max(), pred_arr.max())
plt.xlim(0, 120000)
plt.ylim(0, 120000)
plt.xlabel("Actual Target (Price)")
plt.ylabel("Predicted Value (Price)")
plt.title("予測値と実際のターゲット値のヒートマップ")
plt.legend()
plt.tight_layout()
save_path = os.path.join(runoutput_dir, "03_heatmap_seaborn.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"保存完了: {save_path}")

# --- Graph 4: Seaborn Histplot (修正版ヒートマップ) ---
plt.figure(figsize=(10, 8))
sns.histplot(x=actual_arr, y=pred_arr, bins=50, cmap='coolwarm', cbar=True)
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
plt.xlim(0, 120000)
plt.ylim(0, 120000)
plt.xlabel("Actual Target (Price)")
plt.ylabel("Predicted Value (Price)")
plt.title("予測値と実際のターゲット値のヒートマップ")
plt.legend()
plt.tight_layout()
save_path = os.path.join(runoutput_dir, "03_aheatmap_seaborn.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"保存完了: {save_path}")


# ==========================================
# データ整合性の総合チェック (ID・特徴量・価格)
# ==========================================
print("\n=== データとIDの完全一致確認（ランダム5件） ===")
print("AI学習用データ(復元) と 元データ が一致していれば、IDを消しても紐付けは完璧です。")

# 確認したい特徴量
target_cols = ["築年", "専有・建物面積（平米）"]

# 特徴量リストの何番目にあるか探す（エラー防止付き）
try:
    col_indices = [feature_columns.index(c) for c in target_cols]
except ValueError:
    print("エラー: 指定した列名が feature_columns に見つかりませんでした。列名を確認してください。")
    col_indices = [0, 1] # 仮のインデックス

print("-" * 140)
# ヘッダー作成
header = f"{'Idx':<5} | {'履歴ID':<11} | "
for col in target_cols:
    header += f" {col[:4]:<5} (AI / 元)  | "
header += "    価格 (AI  /  元)    |  判定"
print(header)
print("-" * 140)

# ランダムに5件取得
check_indices = random.sample(range(len(X_train)), 5)

for i in check_indices:
    # --- 1. AIデータ (Tensor -> 逆変換) ---
    # 特徴量を復元 (1行分を逆変換)
    feat_vals_all = scaler_X.inverse_transform(X_train[i].unsqueeze(0).numpy())[0]
    # 指定した特徴量だけ抜き出す
    ai_vals = [feat_vals_all[idx] for idx in col_indices]
    # 価格を復元
    ai_price = scaler_y.inverse_transform(y_train[i].numpy().reshape(-1, 1))[0][0]

    # --- 2. 元データ (DataFrame) ---
    original_idx = idx_train[i] # 紐付けの鍵
    row = data.loc[original_idx]
    
    real_id = row["履歴ID"]
    real_vals = [row[c] for c in target_cols]
    real_price = row[target_column]

    # --- 3. 判定ロジック ---
    
    # --- 3. 判定ロジック ---
    # 【注】ここで「43.7 / 43.6」のように僅かなズレが生じることがあります。
    # 原因: 元データ(float64/倍精度)とAI用データ(float32/単精度)の「桁数の精度の違い」によるものです。
    # 実際には 0.000001 レベルの誤差であり、四捨五入の表示状ズレて見えるだけなので、
    # 学習には全く影響ありません。判定がOKなら無視して大丈夫です。
    # 浮動小数点の誤差を考慮して比較
    
    
    is_vals_match = all(abs(a - r) < 0.1 for a, r in zip(ai_vals, real_vals))
    is_price_match = abs(ai_price - real_price) < 1.0
    
    result = "OK" if (is_vals_match and is_price_match) else "ZURETERU"

    # --- 4. 並べて表示 ---
    # インデックスとID
    out = f"{i:<5} | {real_id:<11} |"
    
    # 特徴量 (築年, 面積...)
    for ai_v, real_v in zip(ai_vals, real_vals):
        out += f" {ai_v:>5.1f} / {real_v:>5.1f}  |"
    
    # 価格と判定
    out += f" {ai_price:>10,.0f} / {real_price:>10,.0f} | {result}"
    
    print(out)

print("-" * 140)
print("\n")