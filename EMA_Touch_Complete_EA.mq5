//+------------------------------------------------------------------+
//|                    EMA_Touch_Complete_EA.mq5                     |
//|              MA角度フィルター＆EMA100損切り機能付き完全版         |
//+------------------------------------------------------------------+
#property copyright "Complete EMA Touch Strategy 2024"
#property version   "4.00"

#include <Trade\Trade.mqh>

//--- 入力パラメータ
input group "=== EMA基本設定 ==="
input int      EMA_Fast = 20;            // 短期EMA（20 or 25）
input int      EMA_Mid = 50;              // 中期EMA
input int      EMA_Slow = 100;            // 長期EMA（70 or 100）

input group "=== MA角度フィルター（画像4準拠）==="
input bool     UseMA20AngleFilter = true; // EMA20角度フィルター使用
input double   MA20MinAngle = 30.0;       // EMA20最小角度（pips/5本）
input bool     UseMA50AngleFilter = true; // EMA50角度フィルター使用  
input double   MA50MinAngle = 20.0;       // EMA50最小角度（pips/5本）
input int      AnglePeriod = 5;           // 角度計算期間（本）

input group "=== ストキャスティクス設定 ==="
input int      Stoch_K = 15;              // %K期間
input int      Stoch_D = 3;               // %D期間
input int      Stoch_Slowing = 9;         // スローイング

input group "=== エントリーフィルター ==="
input bool     SkipFirstTouch = true;     // 最初のタッチをスキップ
input int      MinBarsBetweenTrades = 20; // 最小待機バー数
input double   TouchZone = 2;             // タッチ判定の許容pips

input group "=== 決済設定 ==="
input double   Exit_Stoch_Buy = 80;       // 買い決済ストキャスレベル
input double   Exit_Stoch_Sell = 20;      // 売り決済ストキャスレベル
input double   Alternative_TP_Pips = 50;  // 代替利確pips
input bool     UseEMA100StopLoss = true;  // EMA100損切り使用
input double   Fixed_SL_ATR_Multi = 2.0;  // 固定SL = ATR × この値

input group "=== リスク管理 ==="
input double   RiskPercent = 1.0;         // リスク率（%）
input int      MaxSpread = 30;            // 最大スプレッド
input int      MaxPositions = 1;          // 最大ポジション数

input group "=== その他 ==="
input int      MagicNumber = 20240704;    // マジックナンバー
input string   Comment = "EMA_Complete";  // コメント
input bool     DebugMode = true;          // デバッグモード

//--- グローバル変数
CTrade trade;
int h_ema_fast, h_ema_mid, h_ema_slow, h_stoch, h_atr;

// 状態管理
bool in_perfect_order_up = false;
bool in_perfect_order_down = false;
int touch_count = 0;
datetime last_po_start = 0;
datetime last_trade_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- インジケーターハンドル作成
    h_ema_fast = iMA(_Symbol, _Period, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
    h_ema_mid = iMA(_Symbol, _Period, EMA_Mid, 0, MODE_EMA, PRICE_CLOSE);
    h_ema_slow = iMA(_Symbol, _Period, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
    h_stoch = iStochastic(_Symbol, _Period, Stoch_K, Stoch_D, Stoch_Slowing, MODE_SMA, STO_LOWHIGH);
    h_atr = iATR(_Symbol, _Period, 14);
    
    if(h_ema_fast == INVALID_HANDLE || h_ema_mid == INVALID_HANDLE || 
       h_ema_slow == INVALID_HANDLE || h_stoch == INVALID_HANDLE || h_atr == INVALID_HANDLE)
    {
        Print("インジケーター作成エラー");
        return(INIT_FAILED);
    }
    
    trade.SetExpertMagicNumber(MagicNumber);
    
    Print("=== 完全版EA 初期化完了 ===");
    Print("MA角度フィルター: EMA20=", UseMA20AngleFilter, " EMA50=", UseMA50AngleFilter);
    Print("EMA100損切り: ", UseEMA100StopLoss);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- 新しいバーの確認（確定足でのみ動作）
    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(_Symbol, _Period, 0);
    
    if(last_bar_time == current_bar_time) return;
    last_bar_time = current_bar_time;
    
    //--- EMA100損切りチェック（優先度高）
    if(UseEMA100StopLoss) CheckEMA100StopLoss();
    
    //--- ポジション管理
    ManagePositions();
    
    //--- 既にポジションがある場合は新規エントリーしない
    if(CountPositions() >= MaxPositions) return;
    
    //--- スプレッドチェック
    if(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > MaxSpread) return;
    
    //--- 最小待機時間チェック
    if(current_bar_time - last_trade_time < MinBarsBetweenTrades * PeriodSeconds()) return;
    
    //--- パーフェクトオーダー状態の更新
    UpdatePerfectOrderState();
    
    //--- MA角度チェック（画像4準拠）
    if(!CheckMAAngles()) return;
    
    //--- 買いシグナルチェック
    if(CheckBuySignal())
    {
        ExecuteBuy();
    }
    //--- 売りシグナルチェック
    else if(CheckSellSignal())
    {
        ExecuteSell();
    }
}

//+------------------------------------------------------------------+
//| MA角度チェック（画像4のロジック）                                |
//+------------------------------------------------------------------+
bool CheckMAAngles()
{
    bool angle_ok = true;
    
    //--- EMA20の角度チェック
    if(UseMA20AngleFilter)
    {
        double ma20_angle = CalculateMAAngle(h_ema_fast, "EMA20");
        
        if(in_perfect_order_up && ma20_angle < MA20MinAngle)
        {
            if(DebugMode) Print("EMA20角度不足（上昇）: ", DoubleToString(ma20_angle, 1), " < ", MA20MinAngle);
            angle_ok = false;
        }
        else if(in_perfect_order_down && ma20_angle > -MA20MinAngle)
        {
            if(DebugMode) Print("EMA20角度不足（下降）: ", DoubleToString(ma20_angle, 1), " > ", -MA20MinAngle);
            angle_ok = false;
        }
    }
    
    //--- EMA50の角度チェック
    if(UseMA50AngleFilter && angle_ok)  // EMA20がOKの場合のみチェック
    {
        double ma50_angle = CalculateMAAngle(h_ema_mid, "EMA50");
        
        if(in_perfect_order_up && ma50_angle < MA50MinAngle)
        {
            if(DebugMode) Print("EMA50角度不足（上昇）: ", DoubleToString(ma50_angle, 1), " < ", MA50MinAngle);
            angle_ok = false;
        }
        else if(in_perfect_order_down && ma50_angle > -MA50MinAngle)
        {
            if(DebugMode) Print("EMA50角度不足（下降）: ", DoubleToString(ma50_angle, 1), " > ", -MA50MinAngle);
            angle_ok = false;
        }
    }
    
    return angle_ok;
}

//+------------------------------------------------------------------+
//| MA角度計算（画像4準拠：1本前と5本前の差分）                     |
//+------------------------------------------------------------------+
double CalculateMAAngle(int ma_handle, string ma_name)
{
    double ma_1 = GetEMAValue(ma_handle, 1);     // 1本前
    double ma_past = GetEMAValue(ma_handle, 1 + AnglePeriod);  // 5本前（デフォルト）
    
    // 差分をpips単位で計算
    double angle_pips = (ma_1 - ma_past) / _Point / 10;
    
    if(DebugMode && MathAbs(angle_pips) > 0)
    {
        Print(ma_name, " 角度: ", DoubleToString(angle_pips, 1), " pips/", AnglePeriod, "本");
    }
    
    return angle_pips;
}

//+------------------------------------------------------------------+
//| EMA100損切りチェック                                             |
//+------------------------------------------------------------------+
void CheckEMA100StopLoss()
{
    if(!UseEMA100StopLoss) return;
    
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(!PositionSelectByTicket(PositionGetTicket(i))) continue;
        if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        
        ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        
        // 1本前の確定足の値を取得
        double open_1 = iOpen(_Symbol, _Period, 1);
        double close_1 = iClose(_Symbol, _Period, 1);
        double ema100_1 = GetEMAValue(h_ema_slow, 1);
        
        bool should_close = false;
        
        //--- 買いポジション：実体がEMA100を下抜け
        if(pos_type == POSITION_TYPE_BUY)
        {
            // 実体の上端がEMA100より下
            double body_high = MathMax(open_1, close_1);
            if(body_high < ema100_1)
            {
                should_close = true;
                if(DebugMode) Print("買いポジション：実体がEMA100を下抜け");
            }
        }
        //--- 売りポジション：実体がEMA100を上抜け
        else if(pos_type == POSITION_TYPE_SELL)
        {
            // 実体の下端がEMA100より上
            double body_low = MathMin(open_1, close_1);
            if(body_low > ema100_1)
            {
                should_close = true;
                if(DebugMode) Print("売りポジション：実体がEMA100を上抜け");
            }
        }
        
        if(should_close)
        {
            if(trade.PositionClose(PositionGetTicket(i)))
            {
                Print("EMA100損切り実行");
            }
        }
    }
}

//+------------------------------------------------------------------+
//| パーフェクトオーダー状態の更新                                   |
//+------------------------------------------------------------------+
void UpdatePerfectOrderState()
{
    //--- バッファ取得（1本前の確定足で判定）
    double ema_fast_1 = GetEMAValue(h_ema_fast, 1);
    double ema_mid_1 = GetEMAValue(h_ema_mid, 1);
    double ema_slow_1 = GetEMAValue(h_ema_slow, 1);
    
    //--- 上昇パーフェクトオーダー
    bool po_up = (ema_fast_1 > ema_mid_1 && ema_mid_1 > ema_slow_1);
    
    //--- 下降パーフェクトオーダー
    bool po_down = (ema_fast_1 < ema_mid_1 && ema_mid_1 < ema_slow_1);
    
    //--- 新規パーフェクトオーダー成立
    if(po_up && !in_perfect_order_up)
    {
        in_perfect_order_up = true;
        in_perfect_order_down = false;
        touch_count = 0;
        last_po_start = TimeCurrent();
        
        if(DebugMode) 
        {
            Print("=== 上昇パーフェクトオーダー成立 ===");
            // 角度も表示
            if(UseMA20AngleFilter || UseMA50AngleFilter)
            {
                double ma20_angle = CalculateMAAngle(h_ema_fast, "EMA20");
                double ma50_angle = CalculateMAAngle(h_ema_mid, "EMA50");
            }
        }
    }
    
    if(po_down && !in_perfect_order_down)
    {
        in_perfect_order_down = true;
        in_perfect_order_up = false;
        touch_count = 0;
        last_po_start = TimeCurrent();
        if(DebugMode) Print("=== 下降パーフェクトオーダー成立 ===");
    }
    
    //--- パーフェクトオーダー解除
    if(!po_up) in_perfect_order_up = false;
    if(!po_down) in_perfect_order_down = false;
}

//+------------------------------------------------------------------+
//| 買いシグナルチェック（画像5のロジック）                          |
//+------------------------------------------------------------------+
bool CheckBuySignal()
{
    if(!in_perfect_order_up) return false;
    
    //--- 必要な値を取得
    double low_1 = iLow(_Symbol, _Period, 1);
    double low_2 = iLow(_Symbol, _Period, 2);
    double close_1 = iClose(_Symbol, _Period, 1);
    
    double ema20_1 = GetEMAValue(h_ema_fast, 1);
    double ema50_1 = GetEMAValue(h_ema_mid, 1);
    
    //--- ストキャス値
    double stoch_k_1 = GetStochValue(0, 1);
    double stoch_d_1 = GetStochValue(1, 1);
    double stoch_k_2 = GetStochValue(0, 2);
    double stoch_d_2 = GetStochValue(1, 2);
    
    //--- EMAタッチ判定
    bool ema20_touch = false;
    bool ema50_touch = false;
    
    // EMA20タッチ
    if(low_2 > ema20_1 - TouchZone * _Point * 10 &&
       low_1 < ema20_1 + TouchZone * _Point * 10 &&
       close_1 > ema20_1)
    {
        ema20_touch = true;
        touch_count++;
        if(DebugMode) Print("EMA20タッチ検出（タッチ回数: ", touch_count, "）");
    }
    
    // EMA50タッチ
    if(!ema20_touch &&
       low_2 > ema50_1 - TouchZone * _Point * 10 &&
       low_1 < ema50_1 + TouchZone * _Point * 10 &&
       close_1 > ema50_1)
    {
        ema50_touch = true;
        touch_count++;
        if(DebugMode) Print("EMA50タッチ検出（タッチ回数: ", touch_count, "）");
    }
    
    if(!ema20_touch && !ema50_touch) return false;
    
    //--- 最初のタッチをスキップ
    if(SkipFirstTouch && touch_count == 1)
    {
        if(DebugMode) Print("最初のタッチのためスキップ");
        return false;
    }
    
    //--- ストキャスゴールデンクロス判定
    bool golden_cross = (stoch_k_2 < stoch_d_2 && stoch_k_1 > stoch_d_1);
    
    if(!golden_cross)
    {
        if(DebugMode) Print("ゴールデンクロスなし");
        return false;
    }
    
    //--- MA角度の最終確認
    if(!CheckMAAngles()) return false;
    
    //--- すべての条件を満たした
    if(DebugMode)
    {
        Print("=== 買いシグナル確定 ===");
        Print("タッチEMA: ", ema20_touch ? "EMA20" : "EMA50");
        Print("タッチ回数: ", touch_count);
        Print("Stoch: K=", DoubleToString(stoch_k_1, 1), " D=", DoubleToString(stoch_d_1, 1));
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| 売りシグナルチェック（買いの逆）                                 |
//+------------------------------------------------------------------+
bool CheckSellSignal()
{
    if(!in_perfect_order_down) return false;
    
    //--- 必要な値を取得
    double high_1 = iHigh(_Symbol, _Period, 1);
    double high_2 = iHigh(_Symbol, _Period, 2);
    double close_1 = iClose(_Symbol, _Period, 1);
    
    double ema20_1 = GetEMAValue(h_ema_fast, 1);
    double ema50_1 = GetEMAValue(h_ema_mid, 1);
    
    //--- ストキャス値
    double stoch_k_1 = GetStochValue(0, 1);
    double stoch_d_1 = GetStochValue(1, 1);
    double stoch_k_2 = GetStochValue(0, 2);
    double stoch_d_2 = GetStochValue(1, 2);
    
    //--- EMAタッチ判定
    bool ema20_touch = false;
    bool ema50_touch = false;
    
    // EMA20タッチ
    if(high_2 < ema20_1 + TouchZone * _Point * 10 &&
       high_1 > ema20_1 - TouchZone * _Point * 10 &&
       close_1 < ema20_1)
    {
        ema20_touch = true;
        touch_count++;
    }
    
    // EMA50タッチ
    if(!ema20_touch &&
       high_2 < ema50_1 + TouchZone * _Point * 10 &&
       high_1 > ema50_1 - TouchZone * _Point * 10 &&
       close_1 < ema50_1)
    {
        ema50_touch = true;
        touch_count++;
    }
    
    if(!ema20_touch && !ema50_touch) return false;
    
    //--- 最初のタッチをスキップ
    if(SkipFirstTouch && touch_count == 1) return false;
    
    //--- ストキャスデッドクロス判定
    bool dead_cross = (stoch_k_2 > stoch_d_2 && stoch_k_1 < stoch_d_1);
    
    if(!dead_cross) return false;
    
    //--- MA角度の最終確認
    if(!CheckMAAngles()) return false;
    
    if(DebugMode)
    {
        Print("=== 売りシグナル確定 ===");
        Print("タッチEMA: ", ema20_touch ? "EMA20" : "EMA50");
        Print("タッチ回数: ", touch_count);
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| ポジション管理（決済）                                           |
//+------------------------------------------------------------------+
void ManagePositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(!PositionSelectByTicket(PositionGetTicket(i))) continue;
        if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        
        double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
        double current_price = PositionGetDouble(POSITION_PRICE_CURRENT);
        ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        
        //--- 現在のストキャス値
        double stoch_k = GetStochValue(0, 1);
        
        bool should_close = false;
        string close_reason = "";
        
        //--- 買いポジションの決済
        if(pos_type == POSITION_TYPE_BUY)
        {
            // ストキャス条件
            if(stoch_k >= Exit_Stoch_Buy)
            {
                should_close = true;
                close_reason = "Stoch " + DoubleToString(stoch_k, 1) + " >= " + DoubleToString(Exit_Stoch_Buy, 1);
            }
            
            // 代替利確
            double profit_pips = (current_price - entry_price) / _Point / 10;
            if(profit_pips >= Alternative_TP_Pips)
            {
                should_close = true;
                close_reason = "利確 " + DoubleToString(profit_pips, 1) + " pips";
            }
            
            // パーフェクトオーダー崩壊
            if(!in_perfect_order_up)
            {
                should_close = true;
                close_reason = "パーフェクトオーダー崩壊";
            }
        }
        //--- 売りポジションの決済
        else if(pos_type == POSITION_TYPE_SELL)
        {
            if(stoch_k <= Exit_Stoch_Sell)
            {
                should_close = true;
                close_reason = "Stoch " + DoubleToString(stoch_k, 1) + " <= " + DoubleToString(Exit_Stoch_Sell, 1);
            }
            
            double profit_pips = (entry_price - current_price) / _Point / 10;
            if(profit_pips >= Alternative_TP_Pips)
            {
                should_close = true;
                close_reason = "利確 " + DoubleToString(profit_pips, 1) + " pips";
            }
            
            if(!in_perfect_order_down)
            {
                should_close = true;
                close_reason = "パーフェクトオーダー崩壊";
            }
        }
        
        if(should_close)
        {
            if(trade.PositionClose(PositionGetTicket(i)))
            {
                Print("ポジション決済: ", close_reason);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| 買い注文実行                                                      |
//+------------------------------------------------------------------+
void ExecuteBuy()
{
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double atr = GetATRValue();
    double sl = ask - atr * Fixed_SL_ATR_Multi;
    double tp = 0;  // TPは設定しない（ストキャスで決済）
    
    double lot = CalculateLotSize(ask - sl);
    if(lot <= 0) return;
    
    if(trade.Buy(lot, _Symbol, ask, sl, tp, Comment))
    {
        last_trade_time = iTime(_Symbol, _Period, 0);
        Print("買い注文約定: Price=", ask, " SL=", sl, " Lot=", lot);
        
        // 角度情報も記録
        if(UseMA20AngleFilter || UseMA50AngleFilter)
        {
            double ma20_angle = CalculateMAAngle(h_ema_fast, "EMA20");
            double ma50_angle = CalculateMAAngle(h_ema_mid, "EMA50");
            Print("エントリー時角度 - EMA20: ", DoubleToString(ma20_angle, 1), 
                  " EMA50: ", DoubleToString(ma50_angle, 1));
        }
    }
}

//+------------------------------------------------------------------+
//| 売り注文実行                                                      |
//+------------------------------------------------------------------+
void ExecuteSell()
{
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double atr = GetATRValue();
    double sl = bid + atr * Fixed_SL_ATR_Multi;
    double tp = 0;
    
    double lot = CalculateLotSize(sl - bid);
    if(lot <= 0) return;
    
    if(trade.Sell(lot, _Symbol, bid, sl, tp, Comment))
    {
        last_trade_time = iTime(_Symbol, _Period, 0);
        Print("売り注文約定: Price=", bid, " SL=", sl, " Lot=", lot);
    }
}

//+------------------------------------------------------------------+
//| ヘルパー関数群                                                    |
//+------------------------------------------------------------------+
double GetEMAValue(int handle, int shift)
{
    double buffer[];
    ArraySetAsSeries(buffer, true);
    if(CopyBuffer(handle, 0, shift, 1, buffer) != 1) return 0;
    return buffer[0];
}

double GetStochValue(int buffer_num, int shift)
{
    double buffer[];
    ArraySetAsSeries(buffer, true);
    if(CopyBuffer(h_stoch, buffer_num, shift, 1, buffer) != 1) return 0;
    return buffer[0];
}

double GetATRValue()
{
    double buffer[];
    ArraySetAsSeries(buffer, true);
    if(CopyBuffer(h_atr, 0, 1, 1, buffer) != 1) return 0;
    return buffer[0];
}

double CalculateLotSize(double sl_distance)
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_money = balance * RiskPercent / 100;
    
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    
    double lot = risk_money / (sl_distance / tick_size * tick_value);
    
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lot = MathFloor(lot / lot_step) * lot_step;
    lot = MathMax(min_lot, MathMin(lot, max_lot));
    
    return NormalizeDouble(lot, 2);
}

int CountPositions()
{
    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
               PositionGetString(POSITION_SYMBOL) == _Symbol)
            {
                count++;
            }
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    IndicatorRelease(h_ema_fast);
    IndicatorRelease(h_ema_mid);
    IndicatorRelease(h_ema_slow);
    IndicatorRelease(h_stoch);
    IndicatorRelease(h_atr);
}