import joblib
import pandas as pd # PandasはDataFrame変換試行のために残しておきます

print("学習済みFLAMLモデルを読み込んでいます...")
try:
    automl = joblib.load("/Users/hamadakou/Desktop/trade_log/best_flaml_model.pkl")
    print("モデルの読み込み完了。")
except FileNotFoundError:
    print("エラー: 'best_flaml_model.pkl' が見つかりません。ファイルパスを確認してください。")
    exit() # ファイルがなければ終了
except Exception as e:
    print(f"エラー: モデルの読み込み中に問題が発生しました: {e}")
    exit() # その他のエラーでも終了

# --- リーダーボード形式のデータの取得試行 (以前の試み) ---
print("\n--- リーダーボード形式のデータの取得試行 ---")
leaderboard_df_from_object = None # DataFrameを格納する変数

# 試行1: automl.training_log を確認
# (読み込んだオブジェクトには含まれていない可能性が高いですが、念のためチェック)
if hasattr(automl, 'training_log') and automl.training_log is not None and len(automl.training_log) > 0:
    print("情報を 'automl.training_log' から取得しようとしています...")
    try:
        leaderboard_df_from_object = pd.DataFrame(automl.training_log)
        print("DataFrameへの変換試行: 成功！")
    except Exception as e:
        print(f"エラー: 'automl.training_log' をDataFrameに変換できませんでした: {e}")
        print("生の 'automl.training_log' の内容:", automl.training_log)
else:
    print("'automl.training_log' は見つからないか、空か、内容が期待した形式ではありませんでした。")

# 試行2: automl.results を確認 (training_log がダメだった場合)
if leaderboard_df_from_object is None: # training_log からDataFrameが作れなかった場合のみ試行
    if hasattr(automl, 'results') and automl.results is not None and len(automl.results) > 0:
        print("\n情報を 'automl.results' から取得しようとしています...")
        print("注意: 'automl.results' は複雑な形式の可能性があります。内容をそのまま表示します。")
        print(automl.results)
        print("もし 'automl.results' からリーダーボードを作成したい場合、内容を見て手動で整形する必要があります。")
    else:
        print("'automl.results' は見つからないか、空でした。")

# --- リーダーボードデータの表示/保存 (もし取得できていれば) ---
if leaderboard_df_from_object is not None and not leaderboard_df_from_object.empty:
    print("\n--- 読み込んだモデルオブジェクトから取得できた学習履歴 ---")
    print(leaderboard_df_from_object.to_string()) # 全ての行と列を表示
    try:
        leaderboard_df_from_object.to_csv("flaml_leaderboard_from_object.csv", index=False)
        print("\n✓ 学習履歴を 'flaml_leaderboard_from_object.csv' に保存しました。")
    except Exception as e:
        print(f"\nエラー: 学習履歴をCSVに保存できませんでした: {e}")
else:
    print("\n--- 読み込んだモデルオブジェクトからのリーダーボード取得 ---")
    print("残念ながら、保存されたモデルオブジェクトから直接、詳細なリーダーボードデータを取得できませんでした。")

# --- 読み込まれたモデルの「最良」情報を表示 ---
print("\n--- 読み込まれたモデルの最良情報 ---")
# getattrを使うと、属性が存在しなくてもエラーにならず、デフォルト値を返せます
best_estimator_name = getattr(automl, 'best_estimator', '情報なし')
best_config = getattr(automl, 'best_config', '情報なし')
best_loss = getattr(automl, 'best_loss', '情報なし') # FLAML v1.x, v2.x で利用可能
model_actual = getattr(automl, 'model', None) # 実際のモデルオブジェクト

print(f"最良と判断された学習器の名前: {best_estimator_name}")
print(f"最良モデルの学習時のエラー値 (1-F1スコアの最小値): {best_loss}")
print(f"最良モデルの設定 (ハイパーパラメータ):")
if isinstance(best_config, dict):
    for key, value in best_config.items():
        print(f"  {key}: {value}")
else:
    print(f"  {best_config}")

if model_actual is not None:
    print(f"実際に保存されている最良モデルのオブジェクト: {type(model_actual)}")


# --- 詳細な学習履歴の確認方法について ---
print("\n--- 詳細な学習履歴（リーダーボード）について ---")
print("FLAMLが試行錯誤した各モデルの詳細な結果（リーダーボードのような情報）を確認したい場合は、")
print("学習時に作成されたログファイル 'flaml_longrun.log' をテキストエディタで開いてみてください。")
print("そこには、試行ごとのモデル名、設定（ハイパーパラメータ）、スコアなどが記録されているはずです。")
print("このPythonスクリプトでは、そのログファイルの内容を直接解析していません。")

# --- デバッグ用: automlオブジェクトで利用可能な属性やメソッドの一覧 ---
# 以下の行の先頭の # を削除すると、automlオブジェクトで何が使えるかの一覧が表示されます。
# 問題解決の手がかりになるかもしれません。
# print("\n--- automlオブジェクトで利用可能な属性やメソッドの一覧 (デバッグ用) ---")
# print(dir(automl))

print("\nスクリプトの処理が完了しました。")