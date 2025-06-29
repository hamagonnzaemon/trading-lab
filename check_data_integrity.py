import pandas as pd
import os

# --- このスクリプトをCSVファイルと同じディレクトリに置いて実行してください ---
# --- または、以下のファイルパスを実際の絶対パスに修正してください ---
trades_file = 'trades_ml_roadmap.csv'
bars_file   = 'GOLD_M5_202101040100_202504302355.csv'
ea_csv_file = 'exported_bars_ranged.csv' # 以前共有いただいたEA出力のCSV

# --- ファイル存在チェック ---
for f_path in [trades_file, bars_file, ea_csv_file]:
    if not os.path.exists(f_path):
        print(f"エラー: ファイルが見つかりません: {f_path}")
        print("スクリプトをCSVファイルと同じディレクトリに置くか、ファイルパスを修正してください。")
        exit()

# --- trades_ml_roadmap.csv の読み込み ---
print(f"'{trades_file}' を読み込んでいます...")
trades_df = pd.read_csv(trades_file, parse_dates=['bar_time'])
print(f"  '{trades_file}' から {len(trades_df)} 行を読み込みました。")

# --- GOLD_M5_...csv の読み込みと前処理 ---
print(f"'{bars_file}' を読み込んでいます...")
# GOLD_M5_...csv はタブ区切りであると推測されます (ファイル名と以前のやり取りから)
# もし区切り文字が異なる場合は sep='\t' の部分を修正してください (例: sep=',')
try:
    bars_df = pd.read_csv(bars_file, sep='\t') # タブ区切りを指定
except Exception as e:
    print(f"エラー: '{bars_file}' の読み込みに失敗しました。区切り文字が正しいか確認してください。詳細: {e}")
    exit()

print(f"  '{bars_file}' から {len(bars_df)} 行を読み込みました（生データ）。")
bars_df.columns = [col.strip('<>').lower() for col in bars_df.columns]

required_bars_cols = ['date', 'time', 'open', 'high', 'low', 'close']
missing_cols = [col for col in required_bars_cols if col not in bars_df.columns]
if missing_cols:
    print(f"エラー: バーデータファイル '{bars_file}' に必要なカラムが見つかりません（<>除去・小文字化後）: {missing_cols}")
    exit()

try:
    bars_df['bar_time'] = pd.to_datetime(bars_df['date'] + ' ' + bars_df['time'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
except Exception as e:
    print(f"エラー: 'bar_time' の作成中に問題が発生しました。'date' と 'time' カラムの形式を確認してください。詳細: {e}")
    exit()

bars_df.dropna(subset=['bar_time'], inplace=True) # bar_time のパースに失敗した行を削除
bars_df = bars_df[['bar_time', 'open', 'high', 'low', 'close']]
print(f"  '{bars_file}' の前処理後（bar_time作成・列選択）、{len(bars_df)} 行になりました。")


# --- trades_df と bars_df のマージ ---
print(f"'{trades_file}' と '{bars_file}' を 'bar_time' でマージしています...")
# bar_timeの重複をbars_dfから削除してからマージ (万が一重複がある場合、マージ結果が増えるのを防ぐ)
bars_df.drop_duplicates(subset=['bar_time'], keep='first', inplace=True)
merged_data = pd.merge(trades_df, bars_df, on='bar_time', how='left')
print(f"  マージ後のデータは {len(merged_data)} 行です。")

# --- マージ後データ全体のOHLC NaNチェック ---
print("="*50)
print("マージ後データ全体のOHLC NaNチェック:")
nan_open_count = merged_data['open'].isna().sum()
nan_high_count = merged_data['high'].isna().sum()
nan_low_count = merged_data['low'].isna().sum()
nan_close_count = merged_data['close'].isna().sum()
print(f"  'open' 列のNaN数: {nan_open_count}")
print(f"  'high' 列のNaN数: {nan_high_count}")
print(f"  'low' 列のNaN数:  {nan_low_count}")
print(f"  'close' 列のNaN数: {nan_close_count}")

if nan_close_count > 0:
    print("  'close'がNaNである行の例（全体から5件）:")
    print(merged_data[merged_data['close'].isna()][['bar_time', 'open', 'high', 'low', 'close']].head())
print("="*50)

# --- EAデータとの共通期間におけるOHLC NaNチェック ---
print(f"EAデータ '{ea_csv_file}' を読み込んで時間軸の参考にします...")
ea_df = pd.read_csv(ea_csv_file, parse_dates=["bar_time"])
print(f"  '{ea_csv_file}' から {len(ea_df)} 行を読み込みました。")
if not ea_df.empty:
    print(f"  EAデータの期間: {ea_df['bar_time'].min()} から {ea_df['bar_time'].max()}")

    # EAデータのbar_timeのみを抽出（重複排除）して、merged_dataと左結合
    # これにより、EAに存在する各bar_timeについて、merged_data側のOHLCがどうなっているかを確認
    ea_bar_times_for_check = ea_df[['bar_time']].drop_duplicates()
    common_period_data = pd.merge(ea_bar_times_for_check, merged_data, on='bar_time', how='left')

    print(f"  EAの {len(ea_bar_times_for_check)} 個のユニークなbar_timeに合わせたデータの行数: {len(common_period_data)}")

    print("EAとの共通期間におけるOHLC NaNチェック:")
    nan_open_common = common_period_data['open'].isna().sum()
    nan_high_common = common_period_data['high'].isna().sum()
    nan_low_common = common_period_data['low'].isna().sum()
    nan_close_common = common_period_data['close'].isna().sum()
    print(f"  'open' 列のNaN数 (共通期間): {nan_open_common}")
    print(f"  'high' 列のNaN数 (共通期間): {nan_high_common}")
    print(f"  'low' 列のNaN数 (共通期間):  {nan_low_common}")
    print(f"  'close' 列のNaN数 (共通期間): {nan_close_common}")

    if nan_close_common > 0:
        print("  'close'がNaNである行の例（共通期間から5件）:")
        print(common_period_data[common_period_data['close'].isna()][['bar_time', 'open', 'high', 'low', 'close']].head())
    print("="*50)

    if nan_close_common > 0:
        print("\n【重要】調査結果:")
        print("EAデータと比較している共通の期間内で、Python側のOHLCデータに欠損（NaN）が見つかりました。")
        print("これは、`trades_ml_roadmap.csv` に記載されている一部の `bar_time` に対して、")
        print("`GOLD_M5_202101040100_202504302355.csv` に対応する価格データが存在しないことを意味します。")
        print("これらのNaNの価格データからは正しいテクニカル指標を計算できず、結果として大きな誤差が生じます。")
        print("\n【次のアクション】")
        print("1. `GOLD_M5_...csv` のデータが、`trades_ml_roadmap.csv` の `bar_time` の全期間をカバーしているか確認してください。")
        print("2. もしデータ欠損が意図しないものであれば、価格データを修正または完全なものに置き換えてください。")
        print("3. もし特定の期間のデータが意図的にない場合、その期間の`trades_ml_roadmap.csv`の行を分析対象から除外するか、")
        print("   テクニカル指標計算前にこれらのOHLC欠損行を `step1_feature_enrichment.py` で適切に処理（例: `dropna(subset=['close'])`）する必要があります。")
        print("   ただし、行を削除するとEAのデータと行数が合わなくなる可能性も考慮してください。")
        print("まずは、このOHLCデータの欠損問題を解決することが最優先です。")
    else:
        print("\n【調査結果】:")
        print("EAデータと比較している共通の期間内では、Python側のOHLCデータに欠損（NaN）は見つかりませんでした。")
        print("これは良い兆候です。OHLCデータは比較期間において整合性が取れているようです。")
        print("\n【次の考察ポイント】")
        print("誤差の原因は、以下のいずれかである可能性が高まります。")
        print("  1. テクニカル指標の計算ロジックの微妙な違い（EAの内部計算とPythonライブラリ/カスタム計算との間）。")
        print("  2. Python側で `fillna(0)` を適用したことによる影響（特にEMAなど、初期値の扱いが重要な指標）。")
        print("     EAは初期値を0以外の方法で扱っている可能性があります。")
        print("  3. 浮動小数点演算の精度の問題が積み重なった結果。")
        print("この場合は、各テクニカル指標の計算方法を再度詳細に見直すか、`fillna(0)` の代わりに別の初期値処理（例: EMAなら計算可能な最初の値まで待つ、など）を検討する必要があります。")
else:
    print("EAデータが空または読み込めなかったため、共通期間のNaNチェックはスキップされました。")