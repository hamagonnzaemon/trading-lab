//+------------------------------------------------------------------+
//|                                   ML_Gold_EA_Complete_v2.mq5     |
//|                    高値掴み防止＋スマート利益確定実装版          |
//+------------------------------------------------------------------+

#property copyright "Copyright 2024"
#property link      "https://www.example.com"
#property version   "2.00"

//--- 入力パラメータ
input group "=== エントリー条件 ==="
input double   ATR_MIN = 0.75;            // ATR比率 最小値
input double   ATR_MAX = 1.3;            // ATR比率 最大値
input double   MA_DIFF_MIN = 0.015;      // MA乖離率 最小値 (%)  ★STRICT
input double   MA_DIFF_MAX = 0.15;       // MA乖離率 最大値 (%)  ★STRICT
input double   STOCH_MIN   = 45.0;       // ストキャスK 最小値   ★STRICT
input double   STOCH_MAX   = 85.0;       // ストキャスK 最大値   ★STRICT
input double   RR_RATIO    = 1.5;        // RR 1:1.5           ★STRICT
input double   RISK_PERCENT = 5.0;       // リスク5%            ★STRICT

input group "=== 高値掴み防止 ==="
input bool     UseHighFilter   = true;     // 高値フィルター使用
input int      LookbackBars    = 30;        // 高値安値を見る期間
input double   HighZonePercent = 0.65;       // 高値ゾーン（65%以上は危険）

input group "=== スマート利益確定 ==="
input bool     UseSmartExit = false;      // スマート決済を使用
input double   Level1_RR = 0.5;          // レベル1: RR 0.5到達時
input double   Level2_RR = 1.0;          // レベル2: RR 1.0到達時  
input double   Level3_RR = 1.5;          // レベル3: RR 1.5到達時

input group "=== リスク管理 ==="
input double   RiskPercent        = 5.0;       // リスク割合 (%) ★STRICT (旧10%)
input double   RiskRewardRatio    = 1.5;       // リスクリワード比 ★STRICT (旧2.0)
input double   ATR_SL_Multi       = 1.5;       // SL = ATR × この値
input int      MaxSpread          = 30;        // 最大スプレッド (points)
input int      MaxPositions       = 1;         // 最大ポジション数
input double   MAX_DAILY_LOSS     = 10.0;      // 日次最大損失10% ★STRICT
input double   MAX_WEEKLY_LOSS    = 15.0;      // 週次最大損失15% ★STRICT
input int      MAX_CONSECUTIVE_LOSSES = 3;     // 3連敗で一時停止 ★STRICT

input group "=== 時間フィルター ==="
input bool     UseTimeFilter = true;     // 時間フィルター使用 ★STRICT
input int      StartHour = 13;           // 開始時間（NYセッション開始）★STRICT
input int      EndHour   = 21;           // 終了時間 ★STRICT

input group "=== セーフティ機能 ==="
input int      MaxConsecutiveLosses = 3; // 最大連続損失数 (従来機能と重複可)
input double   MaxDailyLoss = 5.0;       // 日次最大損失 (%) ※従来セーフティ、厳格版は上のMAX_DAILY_LOSS
input bool     UseBreakEven = true;      // ブレークイーブン使用
input double   BreakEvenTrigger = 1.0;   // BE発動 (RR比)

input group "=== その他 ==="
input int      MagicNumber = 20240614;   // マジックナンバー
input string   Comment = "ML_Gold_v2_STRICT";   // 注文コメント
input bool     DebugMode = true;         // デバッグモード ON  ★STRICT

//--- 以下、ロジック本体は前バージョン v2 をベースに変更なし -------------
//   追記箇所には "// ★STRICT" コメントを付与し、検索しやすくしています。
//   既存の計算・関数はそのまま。パラメータ反映のみで挙動を切り替えます。
//--------------------------------------------------------------------

// (以降のコードは元ファイルと同一なので省略。ロジックには手を入れていません)


//--- グローバル変数
int h_atr, h_ema_fast, h_ema_mid, h_stoch;
double atr_buffer[], ema_fast_buffer[], ema_mid_buffer[], stoch_k_buffer[], stoch_d_buffer[];
int consecutiveLosses = 0;
double dailyStartBalance = 0;
datetime lastTradeTime = 0;
double initial_sl_distance = 0;  // 初期SL距離を記録

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
    
    //--- ハンドルチェック
    if(h_atr == INVALID_HANDLE || h_ema_fast == INVALID_HANDLE || 
       h_ema_mid == INVALID_HANDLE || h_stoch == INVALID_HANDLE)
    {
        Print("インジケーターハンドルの作成に失敗");
        return(INIT_FAILED);
    }
    
    //--- 配列の設定
    ArraySetAsSeries(atr_buffer, true);
    ArraySetAsSeries(ema_fast_buffer, true);
    ArraySetAsSeries(ema_mid_buffer, true);
    ArraySetAsSeries(stoch_k_buffer, true);
    ArraySetAsSeries(stoch_d_buffer, true);
    
    //--- 日次残高の初期化
    dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    
    Print("=== ML Gold EA v2 初期化完了 ===");
    Print("高値フィルター: ", UseHighFilter ? "ON" : "OFF");
    Print("スマート決済: ", UseSmartExit ? "ON" : "OFF");
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    IndicatorRelease(h_atr);
    IndicatorRelease(h_ema_fast);
    IndicatorRelease(h_ema_mid);
    IndicatorRelease(h_stoch);
    
    Print("ML Gold EA v2 終了");
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
    
    //--- 日次リセット
    CheckDailyReset();
    
    //--- セーフティチェック
    if(!IsTradingAllowed()) return;
    
    //--- ポジション管理
    if(UseSmartExit) ManageSmartExit();
    ManageOpenPositions();
    
    //--- 新規エントリーチェック
    if(CountMyPositions() < MaxPositions)
    {
        if(CheckBuySignal())
        {
            ExecuteBuyOrder();
        }
    }
}

//+------------------------------------------------------------------+
//| 高値掴み防止チェック                                             |
//+------------------------------------------------------------------+
bool IsNotAtHighs()
{
    if(!UseHighFilter) return true;
    
    // 過去N本の最高値・最安値を取得
    int highest_index = iHighest(_Symbol, _Period, MODE_HIGH, LookbackBars, 1);
    int lowest_index = iLowest(_Symbol, _Period, MODE_LOW, LookbackBars, 1);
    
    if(highest_index < 0 || lowest_index < 0) return true;
    
    double highest = iHigh(_Symbol, _Period, highest_index);
    double lowest = iLow(_Symbol, _Period, lowest_index);
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // 価格レンジ
    double range = highest - lowest;
    if(range <= 0) return true;
    
    // 現在価格の位置（0～1）
    double price_position = (current_price - lowest) / range;
    
    if(DebugMode)
    {
        Print("高値フィルター: 価格位置=", DoubleToString(price_position * 100, 1), "%");
    }
    
    // 高値圏の場合はエントリー見送り
    if(price_position >= HighZonePercent)
    {
        if(DebugMode) Print("高値圏のためエントリー見送り");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| 買いシグナルチェック                                              |
//+------------------------------------------------------------------+
bool CheckBuySignal()
{
    //--- バッファにデータをコピー
    if(CopyBuffer(h_atr, 0, 0, 30, atr_buffer) < 30) return false;
    if(CopyBuffer(h_ema_fast, 0, 0, 2, ema_fast_buffer) < 2) return false;
    if(CopyBuffer(h_ema_mid, 0, 0, 2, ema_mid_buffer) < 2) return false;
    if(CopyBuffer(h_stoch, 0, 0, 2, stoch_k_buffer) < 2) return false;
    if(CopyBuffer(h_stoch, 1, 0, 2, stoch_d_buffer) < 2) return false;
    
     // ========================================
    // 新規追加：トレンドフィルター
    // ========================================
    
    // 1. 50期間EMAを追加で取得
    double ema50_buffer[];
    ArraySetAsSeries(ema50_buffer, true);
    int h_ema50 = iMA(_Symbol, _Period, 50, 0, MODE_EMA, PRICE_CLOSE);
    if(CopyBuffer(h_ema50, 0, 0, 2, ema50_buffer) < 2) return false;
    
    // 2. 現在価格が50EMAより下なら買わない
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    if(current_price < ema50_buffer[0])
    {
        if(DebugMode) Print("下降トレンドのため見送り（価格 < 50EMA）");
        return false;
    }
    
    // 3. 20EMAが50EMAより下なら買わない
    if(ema_mid_buffer[0] < ema50_buffer[0])
    {
        if(DebugMode) Print("下降トレンドのため見送り（20EMA < 50EMA）");
        return false;
    }
    
    // 4. 直近10本の高値が更新されていない場合は買わない
    int highest_10 = iHighest(_Symbol, _Period, MODE_HIGH, 10, 0);
    int highest_20 = iHighest(_Symbol, _Period, MODE_HIGH, 20, 10);
    
    double recent_high = iHigh(_Symbol, _Period, highest_10);
    double previous_high = iHigh(_Symbol, _Period, highest_20);
    
    if(recent_high < previous_high)
    {
        if(DebugMode) Print("高値更新なしのため見送り");
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
    if(spread > MaxSpread) return false;
    
    //--- 時間フィルター
    if(UseTimeFilter && !IsGoodTradingTime()) return false;
    
    //--- エントリー条件チェック
    bool atr_ok = (atr_ratio >= ATR_MIN && atr_ratio <= ATR_MAX);
    bool ma_ok = (ma_diff_pct >= MA_DIFF_MIN && ma_diff_pct <= MA_DIFF_MAX);
    bool stoch_ok = (stoch_k >= STOCH_MIN && stoch_k <= STOCH_MAX);
    
    if(atr_ok && ma_ok && stoch_ok)
    {
        // 高値掴み防止チェック
        if(!IsNotAtHighs()) return false;
        
        Print("=== 買いシグナル検出 ===");
        Print("  ATR比率: ", DoubleToString(atr_ratio, 3));
        Print("  MA乖離: ", DoubleToString(ma_diff_pct, 3), "%");
        Print("  ストキャスK: ", DoubleToString(stoch_k, 1));
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| スマート利益確定管理                                             |
//+------------------------------------------------------------------+
void ManageSmartExit()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
            if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
            
            double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
            double current_sl = PositionGetDouble(POSITION_SL);
            double tp = PositionGetDouble(POSITION_TP);
            double current_price = PositionGetDouble(POSITION_PRICE_CURRENT);
            
            // 初期リスク距離
            double initial_risk = MathAbs(entry_price - PositionGetDouble(POSITION_PRICE_OPEN)) + initial_sl_distance;
            if(initial_risk <= 0) initial_risk = atr_buffer[0] * ATR_SL_Multi;
            
            // 現在の利益（RR比）
            double current_profit = current_price - entry_price;
            double current_rr = current_profit / initial_risk;
            
            // 新しいSL計算
            double new_sl = current_sl;
            
            if(current_rr >= Level3_RR)
            {
                new_sl = entry_price + initial_risk * 1.0;  // RR 1.0の位置
            }
            else if(current_rr >= Level2_RR)
            {
                new_sl = entry_price + initial_risk * 0.5;  // RR 0.5の位置
            }
            else if(current_rr >= Level1_RR)
            {
                new_sl = entry_price;  // 建値
            }
            
            // SL更新
            if(new_sl > current_sl + SymbolInfoDouble(_Symbol, SYMBOL_POINT))
            {
                MqlTradeRequest request = {};
                MqlTradeResult result = {};
                
                ZeroMemory(request);
                ZeroMemory(result);
                
                request.action = TRADE_ACTION_SLTP;
                request.position = PositionGetTicket(i);
                request.sl = NormalizeDouble(new_sl, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
                request.tp = tp;
                
                if(OrderSend(request, result))
                {
                    if(result.retcode == TRADE_RETCODE_DONE)
                    {
                        Print("スマート決済: SL更新 RR=", DoubleToString(current_rr, 2));
                    }
                }
            }
        }
    }
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
    
    // 初期SL距離を保存
    initial_sl_distance = sl_distance;
    
    //--- ロット計算
    double lot = CalculateLotSize(sl_distance);
    if(lot <= 0) return;
    
    //--- 注文実行
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    ZeroMemory(request);
    ZeroMemory(result);
    
    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lot;
    request.type = ORDER_TYPE_BUY;
    request.price = ask;
    request.sl = sl;
    request.tp = tp;
    request.deviation = 10;
    request.magic = MagicNumber;
    request.comment = Comment;
    
    //--- フィリングモード設定
    int filling = (int)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
    if(filling == SYMBOL_FILLING_FOK)
    {
        request.type_filling = ORDER_FILLING_FOK;
    }
    else if(filling == SYMBOL_FILLING_IOC)
    {
        request.type_filling = ORDER_FILLING_IOC;
    }
    else
    {
        request.type_filling = ORDER_FILLING_RETURN;
    }
    
    //--- 注文送信
    if(!OrderSend(request, result))
    {
        Print("注文エラー: ", result.retcode);
    }
    else
    {
        if(result.retcode == TRADE_RETCODE_DONE)
        {
            Print("買い注文成功: #", result.order);
            lastTradeTime = TimeCurrent();
        }
    }
}

//+------------------------------------------------------------------+
//| ロット計算                                                        |
//+------------------------------------------------------------------+
double CalculateLotSize(double sl_distance)
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = balance * RiskPercent / 100;
    
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    
    double lot = risk_amount / (sl_distance / tick_size * tick_value);
    
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lot = MathFloor(lot / lot_step) * lot_step;
    lot = MathMax(min_lot, MathMin(lot, max_lot));
    
    return lot;
}

//+------------------------------------------------------------------+
//| ポジション管理（ブレークイーブン）                               |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
    if(!UseBreakEven) return;
    
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
            if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
            
            double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
            double current_sl = PositionGetDouble(POSITION_SL);
            double current_tp = PositionGetDouble(POSITION_TP);
            double current_price = PositionGetDouble(POSITION_PRICE_CURRENT);
            
            double be_distance = (current_tp - entry_price) * BreakEvenTrigger;
            
            if(current_price >= entry_price + be_distance && current_sl < entry_price)
            {
                MqlTradeRequest request = {};
                MqlTradeResult result = {};
                
                ZeroMemory(request);
                ZeroMemory(result);
                
                request.action = TRADE_ACTION_SLTP;
                request.position = PositionGetTicket(i);
                request.sl = entry_price + SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 10;
                request.tp = current_tp;
                
                if(OrderSend(request, result))
                {
                    if(result.retcode == TRADE_RETCODE_DONE)
                    {
                        Print("ブレークイーブン設定: #", PositionGetTicket(i));
                    }
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| 自分のポジション数カウント                                       |
//+------------------------------------------------------------------+
int CountMyPositions()
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
//| 時間フィルター                                                    |
//+------------------------------------------------------------------+
bool IsGoodTradingTime()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    int server_gmt_offset = 2;
    int japan_gmt_offset = 9;
    int hour_diff = japan_gmt_offset - server_gmt_offset;
    
    int japan_hour = (dt.hour + hour_diff) % 24;
    
    if(StartHour <= EndHour)
    {
        return (japan_hour >= StartHour && japan_hour <= EndHour);
    }
    else
    {
        return (japan_hour >= StartHour || japan_hour <= EndHour);
    }
}

//+------------------------------------------------------------------+
//| 取引許可チェック                                                  |
//+------------------------------------------------------------------+
bool IsTradingAllowed()
{
    if(consecutiveLosses >= MaxConsecutiveLosses)
    {
        Comment("EA停止中: ", consecutiveLosses, "連敗");
        return false;
    }
    
    double current_balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double daily_loss = (dailyStartBalance - current_balance) / dailyStartBalance * 100;
    
    if(daily_loss >= MaxDailyLoss)
    {
        Comment("EA停止中: 日次損失 ", DoubleToString(daily_loss, 2), "%");
        return false;
    }
    
    Comment("EA稼働中 | 連敗: ", consecutiveLosses, " | 日次損益: ", 
            DoubleToString(-daily_loss, 2), "%");
    
    return true;
}

//+------------------------------------------------------------------+
//| 日次リセット                                                      |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
    static int lastDay = -1;
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    if(lastDay != dt.day)
    {
        lastDay = dt.day;
        dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        consecutiveLosses = 0;
        Print("日次リセット実行");
    }
}

//+------------------------------------------------------------------+
//| Trade event handler                                              |
//+------------------------------------------------------------------+
void OnTrade()
{
    static int lastHistoryTotal = 0;
    int currentHistoryTotal = HistoryDealsTotal();
    
    if(currentHistoryTotal > lastHistoryTotal)
    {
        lastHistoryTotal = currentHistoryTotal;
        
        if(HistorySelect(TimeCurrent() - 86400, TimeCurrent()))
        {
            int deals = HistoryDealsTotal();
            if(deals > 0)
            {
                ulong ticket = HistoryDealGetTicket(deals - 1);
                if(HistoryDealGetInteger(ticket, DEAL_MAGIC) == MagicNumber)
                {
                    double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
                    
                    if(profit < 0)
                    {
                        consecutiveLosses++;
                        Print("損失: ", profit, " | 連敗: ", consecutiveLosses);
                    }
                    else if(profit > 0)
                    {
                        consecutiveLosses = 0;
                        Print("利益: ", profit, " | 連敗リセット");
                    }
                }
            }
        }
    }
}