import os
import chardet
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# =========================
# 設定
# =========================
INPUT_FILE = "5_2022京都賃貸データ.xlsx"
OUTPUT_FILE = "tesutoデータ変更5_2022京都賃貸データ.xlsx"

ONE_HOT_COLUMNS = ["建物構造", "所在地名1"]
LABEL_ENCODE_COLUMNS = ["所在地名1", "所在地名2", "沿線名称", "駅名称"]

TARGET_KEY_COLUMN = "間取りタイプ"
TARGET_VALUE_COLUMN = "物件賃料・価格（円）"


# =========================
# ユーティリティ
# =========================
def detect_encoding(sample_path: str, sample_bytes: int = 100000) -> dict:
    """ファイル先頭から文字コードを推定する。"""
    with open(sample_path, "rb") as f:
        raw = f.read(sample_bytes)
    return chardet.detect(raw)


def load_table(input_file: str) -> pd.DataFrame:
    """Excel/CSV を読み込む。CSV は複数エンコーディングを試行する。"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"入力ファイルが見つかりません: {input_file} (cwd={os.getcwd()})"
        )

    if input_file.lower().endswith((".xls", ".xlsx")):
        try:
            data = pd.read_excel(input_file, engine="openpyxl")
            print("read_excel OK")
            return data
        except Exception as e:
            print("read_excel エラー:", e)
            raise

    guess = detect_encoding(input_file)
    print("chardet guess:", guess)
    enc_candidates = []
    if guess and guess.get("encoding"):
        enc_candidates.append(guess["encoding"])
    enc_candidates += ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"]

    tried = set()
    last_exc = None
    for enc in enc_candidates:
        if enc in tried:
            continue
        tried.add(enc)
        try:
            data = pd.read_csv(input_file, encoding=enc, low_memory=False)
            print(f"read_csv success with encoding={enc}")
            return data
        except Exception as e:
            last_exc = e
            print(f"read_csv failed with encoding={enc}: {e}")

    try:
        data = pd.read_csv(input_file, encoding="latin1", low_memory=False)
        print("read_csv fallback with latin1 (may be garbled)")
        return data
    except Exception:
        print("すべての読み込みに失敗しました。エラーメッセージを確認してください。")
        raise last_exc


def print_columns(data: pd.DataFrame) -> None:
    """列名一覧を出力する。"""
    print("データフレームの列名:")
    for column in data.columns:
        print(f"- {column}")


def add_equipment_count(data: pd.DataFrame) -> None:
    """設備列を自動取得し、設備数を計算して追加する。"""
    equipment_columns = [col for col in data.columns if "設備" in col]
    if not equipment_columns:
        data["設備数"] = 0
        return

    equipment_flags = data[equipment_columns].apply(
        lambda series: series.map(
            lambda x: 1 if pd.notnull(x) and str(x).strip() != "" else 0
        )
    )
    data["設備数"] = equipment_flags.sum(axis=1)


def apply_one_hot(data: pd.DataFrame, columns_to_onehot: list) -> None:
    """One-hot エンコーディングを適用する。"""
    for column in columns_to_onehot:
        if column in data.columns:
            one_hot = pd.get_dummies(data[column], prefix=column).astype(int)
            data[one_hot.columns] = one_hot


def apply_target_encoding(data: pd.DataFrame) -> None:
    """間取りタイプにターゲットエンコーディングを付与する。"""
    if TARGET_KEY_COLUMN in data.columns and TARGET_VALUE_COLUMN in data.columns:
        target_mean = data.groupby(TARGET_KEY_COLUMN)[TARGET_VALUE_COLUMN].mean()
        target_median = data.groupby(TARGET_KEY_COLUMN)[TARGET_VALUE_COLUMN].median()

        data[f"{TARGET_KEY_COLUMN}_ターゲットエンコーディング_平均"] = data[
            TARGET_KEY_COLUMN
        ].map(target_mean).round()
        data[f"{TARGET_KEY_COLUMN}_ターゲットエンコーディング_中央値"] = data[
            TARGET_KEY_COLUMN
        ].map(target_median).round()

        print("\n間取りタイプのターゲットエンコーディング結果（平均値と中央値）:")
        print(
            data[
                [
                    TARGET_KEY_COLUMN,
                    f"{TARGET_KEY_COLUMN}_ターゲットエンコーディング_平均",
                    f"{TARGET_KEY_COLUMN}_ターゲットエンコーディング_中央値",
                ]
            ].head()
        )
    else:
        print("\n間取りタイプまたはターゲット列がデータに存在しません。")


def print_target_encoding_summary(data: pd.DataFrame) -> None:
    """間取りタイプごとのターゲットエンコーディング値を出力する。"""
    if TARGET_KEY_COLUMN in data.columns and TARGET_VALUE_COLUMN in data.columns:
        print("\n間取りタイプごとのターゲットエンコーディング値（平均値と中央値）:")
        target_mean = data.groupby(TARGET_KEY_COLUMN)[TARGET_VALUE_COLUMN].mean()
        target_median = data.groupby(TARGET_KEY_COLUMN)[TARGET_VALUE_COLUMN].median()
        for room_type in target_mean.index:
            print(
                f"間取りタイプ: {room_type} → 平均ターゲット値: {target_mean[room_type]}, "
                f"中央ターゲット値: {target_median[room_type]}"
            )
    else:
        print("\n間取りタイプまたはターゲット列がデータに存在しません。")


def fill_missing_values(data: pd.DataFrame) -> None:
    """欠損値の簡易補完を行う。"""
    if "共益費（円）" in data.columns:
        data["共益費（円）"] = data["共益費（円）"].fillna(0)

    if "管理費（円）" in data.columns:
        data["管理費（円）"] = data["管理費（円）"].fillna(0)

    for col in ["徒歩(分)", "バス(分)"]:
        if col in data.columns:
            data[col] = data[col].fillna(0)
            print(f"{col} の欠損値を0で埋めました。")
        else:
            print(f"{col} 列がデータに存在しません。")


def apply_label_encoding(data: pd.DataFrame, columns_to_labelencode: list) -> None:
    """ラベルエンコーディングを適用する。"""
    label_encoder = LabelEncoder()
    for column in columns_to_labelencode:
        if column in data.columns:
            data[column] = data[column].fillna("0").astype(str)
            data[f"{column}_数値"] = label_encoder.fit_transform(data[column])


def print_uniques(data: pd.DataFrame, column: str, title: str) -> None:
    """指定列のユニーク値を出力する。"""
    if column in data.columns:
        unique_values = data[column].dropna().unique()
        print(f"\n{title}:")
        for value in unique_values:
            print(f"- {value}")
    else:
        print(f"\n{column}列がデータに存在しません。")


def main() -> None:
    """前処理パイプラインの実行。"""
    data = load_table(INPUT_FILE)
    print_columns(data)

    add_equipment_count(data)
    apply_one_hot(data, ONE_HOT_COLUMNS)
    apply_target_encoding(data)
    print_target_encoding_summary(data)
    fill_missing_values(data)
    apply_label_encoding(data, LABEL_ENCODE_COLUMNS)

    print("\n処理後のデータ:")
    print(data.head())

    data.to_excel(OUTPUT_FILE, index=False)
    print(f"処理済みデータを '{OUTPUT_FILE}' に保存しました。")

    print("データフレームの列名:")
    print(data.columns)
    print_uniques(data, "建物構造", "建物構造の種類")
    print_uniques(data, "所在地名1", "所在地名1の種類")


if __name__ == "__main__":
    main()