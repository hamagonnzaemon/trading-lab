import joblib, json
model = joblib.load("/Users/hamadakou/Desktop/trade_log/classweight_model.pkl")

# --- 汎用的な情報 ---
print("Estimator type:", type(model))
print("Params:", model.get_params() if hasattr(model, "get_params") else "N/A")

# --- LightGBM sklearn ラッパーだった場合 ---
if hasattr(model, "booster_"):
    booster = model.booster_
    print("Feature names:", booster.feature_name())
    print("Num trees:", booster.num_trees())
    print("Best iteration:", booster.best_iteration)

    # 1) 重要度
    imp_gain = booster.feature_importance(importance_type="gain")
    for f, g in sorted(zip(booster.feature_name(), imp_gain), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{f:20s}: {g}")

    # 2) 全ツリー構造を JSON で保存
    with open("model_dump.json", "w") as fp:
        json.dump(booster.dump_model(), fp, indent=2)
    print("Full model dumped to model_dump.json")
