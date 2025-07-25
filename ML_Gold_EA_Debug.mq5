//+------------------------------------------------------------------+
//|                                      ML_Gold_EA_Debug.mq5        |
//|                              Debug Version for Testing           |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024"
#property link      "https://www.example.com"
#property version   "1.00"

//--- 入力パラメータ（緩めに設定）
input group "=== エントリー条件（デバッグ用に緩和） ==="
input double   ATR_MIN = 0.6;            // ATR比率 最小値（緩和）
input double   ATR_MAX = 1.5;            // ATR比率 最大値（緩和）
input double   MA_DIFF_MIN = 0.0;        // MA乖離率 最小値 (%)（緩和）
input double   MA_DIFF_MAX = 0.2;        // MA乖離率 最大値 (%)（緩和）
input double   STOCH_MIN = 40.0;         // ストキャスK 最小値（緩和）
input double   STOCH_MAX = 90.0;         // ストキャスK 最大値（緩和）

input group "=== リスク管理 ==="
input double   RiskPercent = 1.0;        // リスク割合 (%)
input double   RiskRewardRatio = 2.0;    // リスクリワード比
input double   ATR_SL_Multi = 1.5;       // SL = ATR × この値
input int      MaxSpread = 50;           // 最大スプレッド (points)（緩和）

input group "=== デバッグ ==="
input bool     DebugMode = true;         // デバッグモード
input bool     UseTimeFilter = false;    // 時間フィルター使用（無効化）

//--- グローバル変数
int h_atr, h_ema_fast, h_ema_mid, h_stoch;
double atr_buffer[], ema_fast_buffer[], ema_mid_buffer[], stoch_k_buffer[];
int debugCounter = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- インジケーターハンドルの作成
    h_atr = iATR(_Symbol, _Period, 14);
    h_ema_fast = iMA(_Symbol, _Period, 7, 0, MODE_EMA, PRICE_CLOSE);
    h_ema_mid = iMA(_Symbol, _Period, 20, 0, MODE_EMA, PRICE_CLOSE);
    h_stoch = iStochastic(_Symbol, _Period, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
    
    if(h_atr == INVALID_HANDLE || h_ema_fast == INVALID_HANDLE || 
       h_ema_mid == INVALID_HANDLE || h_stoch == INVALID_HANDLE)
    {
        Print("ERROR: インジケーターハンドルの作成に失敗");
        return(INIT_FAILED);
    }
    
    ArraySetAsSeries(atr_buffer, true);
    ArraySetAsSeries(ema_fast_buffer, true);
    ArraySetAsSeries(ema_mid_buffer, true);
    ArraySetAsSeries(stoch_k_buffer, true);
    
    Print("=== ML Gold EA Debug 初期化完了 ===");
    Print("Symbol: ", _Symbol);
    Print("Period: ", _Period);
    Print("ATR Handle: ", h_atr);
    Print("EMA Fast Handle: ", h_ema_fast);
    Print("EMA Mid Handle: ", h_ema_mid);
    Print("Stoch Handle: ", h_stoch);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- 新しいバーのチェック
    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, _Period, 0);
    if(lastBarTime == currentBarTime) return;
    lastBarTime = currentBarTime;
    
    //--- デバッグカウンター
    debugCounter++;
    if(DebugMode && debugCounter % 10 == 0)  // 10バーごとにデバッグ出力
    {
        DebugPrintValues();
    }
    
    //--- エントリーチェック
    if(CheckBuySignal())
    {
        Print("★★★ 買いシグナル検出！ ★★★");
        ExecuteBuyOrder();
    }
}

//+------------------------------------------------------------------+
//| デバッグ用の値出力                                                |
//+------------------------------------------------------------------+
void DebugPrintValues()
{
    //--- バッファにデータをコピー
    int atr_copied = CopyBuffer(h_atr, 0, 0, 30, atr_buffer);
    int ema_fast_copied = CopyBuffer(h_ema_fast, 0, 0, 2, ema_fast_buffer);
    int ema_mid_copied = CopyBuffer(h_ema_mid, 0, 0, 2, ema_mid_buffer);
    int stoch_copied = CopyBuffer(h_stoch, 0, 0, 2, stoch_k_buffer);
    
    Print("=== デバッグ出力 (Bar #", debugCounter, ") ===");
    Print("コピーしたバッファ数: ATR=", atr_copied, ", EMA_Fast=", ema_fast_copied, 
          ", EMA_Mid=", ema_mid_copied, ", Stoch=", stoch_copied);
    
    if(atr_copied < 30 || ema_fast_copied < 2 || ema_mid_copied < 2 || stoch_copied < 2)
    {
        Print("ERROR: バッファコピーに失敗");
        return;
    }
    
    //--- ATR比率の計算
    double atr_sum = 0;
    for(int i = 0; i < 30; i++)
    {
        atr_sum += atr_buffer[i];
    }
    double atr_ma30 = atr_sum / 30;
    double atr_ratio = atr_buffer[0] / atr_ma30;
    
    //--- MA乖離率の計算
    double ma_diff_pct = (ema_fast_buffer[0] - ema_mid_buffer[0]) / ema_mid_buffer[0] * 100;
    
    //--- ストキャスティクス
    double stoch_k = stoch_k_buffer[0];
    
    //--- 値の出力
    Print("現在値:");
    Print("  ATR[0]: ", atr_buffer[0]);
    Print("  ATR_MA30: ", atr_ma30);
    Print("  ATR比率: ", DoubleToString(atr_ratio, 3), " [範囲: ", ATR_MIN, " - ", ATR_MAX, "]");
    Print("  EMA_Fast: ", ema_fast_buffer[0]);
    Print("  EMA_Mid: ", ema_mid_buffer[0]);
    Print("  MA乖離率: ", DoubleToString(ma_diff_pct, 3), "% [範囲: ", MA_DIFF_MIN, " - ", MA_DIFF_MAX, "]");
    Print("  Stoch_K: ", DoubleToString(stoch_k, 1), " [範囲: ", STOCH_MIN, " - ", STOCH_MAX, "]");
    
    //--- 条件チェック
    bool atr_ok = (atr_ratio >= ATR_MIN && atr_ratio <= ATR_MAX);
    bool ma_ok = (ma_diff_pct >= MA_DIFF_MIN && ma_diff_pct <= MA_DIFF_MAX);
    bool stoch_ok = (stoch_k >= STOCH_MIN && stoch_k <= STOCH_MAX);
    
    Print("条件判定:");
    Print("  ATR条件: ", atr_ok ? "OK" : "NG");
    Print("  MA条件: ", ma_ok ? "OK" : "NG");
    Print("  Stoch条件: ", stoch_ok ? "OK" : "NG");
    Print("  全条件: ", (atr_ok && ma_ok && stoch_ok) ? "★OK★" : "NG");
}

//+------------------------------------------------------------------+
//| 買いシグナルチェック                                              |
//+------------------------------------------------------------------+
bool CheckBuySignal()
{
    //--- バッファにデータをコピー
    if(CopyBuffer(h_atr, 0, 0, 30, atr_buffer) < 30) 
    {
        if(DebugMode) Print("ERROR: ATRバッファコピー失敗");
        return false;
    }
    if(CopyBuffer(h_ema_fast, 0, 0, 2, ema_fast_buffer) < 2)
    {
        if(DebugMode) Print("ERROR: EMA Fastバッファコピー失敗");
        return false;
    }
    if(CopyBuffer(h_ema_mid, 0, 0, 2, ema_mid_buffer) < 2)
    {
        if(DebugMode) Print("ERROR: EMA Midバッファコピー失敗");
        return false;
    }
    if(CopyBuffer(h_stoch, 0, 0, 2, stoch_k_buffer) < 2)
    {
        if(DebugMode) Print("ERROR: Stochバッファコピー失敗");
        return false;
    }
    
    //--- ATR比率の計算
    double atr_sum = 0;
    for(int i = 0; i < 30; i++)
    {
        atr_sum += atr_buffer[i];
    }
    double atr_ma30 = atr_sum / 30;
    double atr_ratio = atr_buffer[0] / atr_ma30;
    
    //--- MA乖離率の計算
    double ma_diff_pct = (ema_fast_buffer[0] - ema_mid_buffer[0]) / ema_mid_buffer[0] * 100;
    
    //--- ストキャスティクス
    double stoch_k = stoch_k_buffer[0];
    
    //--- スプレッドチェック
    long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    if(spread > MaxSpread) 
    {
        if(DebugMode) Print("スプレッドが大きすぎる: ", spread);
        return false;
    }
    
    //--- エントリー条件チェック
    bool atr_ok = (atr_ratio >= ATR_MIN && atr_ratio <= ATR_MAX);
    bool ma_ok = (ma_diff_pct >= MA_DIFF_MIN && ma_diff_pct <= MA_DIFF_MAX);
    bool stoch_ok = (stoch_k >= STOCH_MIN && stoch_k <= STOCH_MAX);
    
    if(atr_ok && ma_ok && stoch_ok)
    {
        Print("=== 買いシグナル詳細 ===");
        Print("  ATR比率: ", DoubleToString(atr_ratio, 3));
        Print("  MA乖離: ", DoubleToString(ma_diff_pct, 3), "%");
        Print("  ストキャスK: ", DoubleToString(stoch_k, 1));
        Print("  スプレッド: ", spread);
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| 買い注文実行                                                      |
//+------------------------------------------------------------------+
void ExecuteBuyOrder()
{
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    
    //--- SL/TP計算
    double atr = atr_buffer[0];
    double sl_distance = atr * ATR_SL_Multi;
    double tp_distance = sl_distance * RiskRewardRatio;
    
    double sl = NormalizeDouble(bid - sl_distance, digits);
    double tp = NormalizeDouble(bid + tp_distance, digits);
    
    //--- ロット計算（固定0.01ロット）
    double lot = 0.01;
    
    //--- 注文実行
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lot;
    request.type = ORDER_TYPE_BUY;
    request.price = ask;
    request.sl = sl;
    request.tp = tp;
    request.deviation = 10;
    request.magic = 12345;
    request.comment = "Debug";
    
    if(OrderSend(request, result))
    {
        if(result.retcode == TRADE_RETCODE_DONE)
        {
            Print("買い注文成功: #", result.order);
        }
        else
        {
            Print("注文失敗 retcode: ", result.retcode);
        }
    }
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    IndicatorRelease(h_atr);
    IndicatorRelease(h_ema_fast);
    IndicatorRelease(h_ema_mid);
    IndicatorRelease(h_stoch);
}