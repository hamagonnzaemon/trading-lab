import sys
import os

print("--- 今、実行されているPythonの場所 ---")
print(sys.executable)
print("\n" + "="*30 + "\n")

print("--- Pythonが部品(ライブラリ)を探しに行く場所のリスト ---")
for path in sys.path:
    print(path)
print("\n" + "="*30 + "\n")


# plyer がインストールされているはずの場所をフルパスで確認
plyer_expected_path = os.path.abspath('myenv/lib/python3.13/site-packages')

print(f"--- plyer が本来あるべき場所 ---\n{plyer_expected_path}\n")

# 本来あるべき場所が、探しに行く場所のリストに含まれているかチェック
if plyer_expected_path in sys.path:
    print("【診断結果】\nリストには含まれています。原因は他にあるかもしれません。")
else:
    print("【診断結果】\n問題を発見しました！\nPythonが探しに行く場所のリストに、plyerのあるべき場所が含まれていません。")