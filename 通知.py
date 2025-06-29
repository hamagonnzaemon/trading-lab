# 必要な部品（notification）をplyerから取り込みます
from plyer import notification

# 通知を送るための設定をします
notification.notify(
    title='ここがタイトルです',
    message='ここにメッセージ本文が入ります。',
    app_name='アプリの名前'
    # timeout=10  # 10秒後に通知を自動的に消したい場合
)