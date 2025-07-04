//+------------------------------------------------------------------+
//|                                           DowTheoryEA_Pro.mq5    |
//|                              改良版ダウ理論トレードEA            |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property link      ""
#property version   "2.00"
#property strict

// 入力パラメータ - 基本設定
input group "=== 基本設定 ==="
input double   LotSize = 0.01;           // ロットサイズ
input int      StopLoss = 100;           // ストップロス (pips)
input int      TakeProfit = 200;         // テイクプロフィット (pips)
input int      MagicNumber = 12345;      // マジックナンバー

// 入力パラメータ - ダウ理論設定
input group "=== ダウ理論設定 ==="
input int      SwingPeriod = 10;         // スイング検出期間
input int      MinSwingSize = 20;        // 最小スイングサイズ (pips)
input bool     RequireThreeSwings = true;// 3つのスイングで確認
input double   RetracementLevel = 0.382; // フィボナッチリトレースメントレベル

// 入力パラメータ - フィルター設定
input group "=== フィルター設定 ==="
input bool     UseMAFilter = true;       // 移動平均フィルター使用
input int      MAPeriod = 200;           // 移動平均期間
input bool     UseMomentumFilter = true; // モメンタムフィルター使用
input int      RSIPeriod = 14;           // RSI期間
input double   RSIOverbought = 70;       // RSI買われすぎレベル
input double   RSIOversold = 30;         // RSI売られすぎレベル

// 入力パラメータ - リスク管理
input group "=== リスク管理 ==="
input bool     UseTrailingStop = true;   // トレーリングストップ使用
input int      TrailingStop = 50;        // トレーリングストップ (pips)
input int      TrailingStep = 10;        // トレーリングステップ (pips)
input bool     UseBreakeven = true;      // ブレークイーブン使用
input int      BreakevenPips = 30;       // ブレークイーブン開始 (pips)
input int      BreakevenProfit = 5;      // ブレークイーブン利益 (pips)
input int      MaxPositions = 1;         // 最大ポジション数

// 入力パラメータ - 時間フィルター
input group "=== 時間フィルター ==="
input bool     UseTimeFilter = false;    // 時間フィルター使用
input int      StartHour = 8;            // 取引開始時間
input int      EndHour = 20;             // 取引終了時間

// 入力パラメータ - デバッグ
input group "=== デバッグ ==="
input bool     ShowDebugInfo = false;    // デバッグ情報表示

// グローバル変数
double point;
int digits;
int maHandle, rsiHandle;

// スイングポイント構造体
struct SwingPoint
{
   double price;
   datetime time;
   int type; // 1: High, -1: Low
   int barIndex;
};

SwingPoint swingHighs[];
SwingPoint swingLows[];
int currentTrend = 0;
datetime lastTradeTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // 通貨ペアの情報を取得
   digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   // インジケーターハンドルの作成
   if(UseMAFilter)
   {
      maHandle = iMA(_Symbol, PERIOD_CURRENT, MAPeriod, 0, MODE_SMA, PRICE_CLOSE);
      if(maHandle == INVALID_HANDLE)
      {
         Print("移動平均の作成に失敗しました");
         return(INIT_FAILED);
      }
   }
   
   if(UseMomentumFilter)
   {
      rsiHandle = iRSI(_Symbol, PERIOD_CURRENT, RSIPeriod, PRICE_CLOSE);
      if(rsiHandle == INVALID_HANDLE)
      {
         Print("RSIの作成に失敗しました");
         return(INIT_FAILED);
      }
   }
   
   // 配列の初期化
   ArrayResize(swingHighs, 0);
   ArrayResize(swingLows, 0);
   
   Print("EA初期化完了: ", _Symbol, " SwingPeriod=", SwingPeriod);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // インジケーターハンドルの解放
   if(maHandle != INVALID_HANDLE) IndicatorRelease(maHandle);
   if(rsiHandle != INVALID_HANDLE) IndicatorRelease(rsiHandle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // 新しいバーの形成時のみ処理
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   if(lastBarTime == currentBarTime)
      return;
      
   lastBarTime = currentBarTime;
   
   // 時間フィルターチェック
   if(UseTimeFilter && !IsTradeTime())
      return;
   
   // スイングポイントの更新
   DetectSwingPoints();
   
   // トレンドの判定
   int newTrend = DetermineTrend();
   
   // ポジション管理
   ManagePositions();
   
   // エントリーシグナルのチェック
   if(GetPositionCount() < MaxPositions && TimeCurrent() - lastTradeTime > 300) // 5分のクールダウン
   {
      int signal = GetEntrySignal(newTrend);
      
      if(signal == 1)
      {
         if(ShowDebugInfo) Print("買いシグナル検出");
         OpenBuyOrder();
         lastTradeTime = TimeCurrent();
      }
      else if(signal == -1)
      {
         if(ShowDebugInfo) Print("売りシグナル検出");
         OpenSellOrder();
         lastTradeTime = TimeCurrent();
      }
   }
   
   currentTrend = newTrend;
}

//+------------------------------------------------------------------+
//| エントリーシグナルを取得する関数                                |
//+------------------------------------------------------------------+
int GetEntrySignal(int trend)
{
   // トレンド転換チェック
   bool trendChange = (trend == 1 && currentTrend <= 0) || (trend == -1 && currentTrend >= 0);
   if(!trendChange || trend == 0)
      return 0;
   
   // MAフィルターチェック
   if(UseMAFilter)
   {
      double ma[];
      ArraySetAsSeries(ma, true);
      if(CopyBuffer(maHandle, 0, 0, 1, ma) <= 0)
         return 0;
      
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      if(trend == 1 && price < ma[0]) return 0;  // 買いシグナルで価格がMA以下
      if(trend == -1 && price > ma[0]) return 0; // 売りシグナルで価格がMA以上
   }
   
   // RSIフィルターチェック
   if(UseMomentumFilter)
   {
      double rsi[];
      ArraySetAsSeries(rsi, true);
      if(CopyBuffer(rsiHandle, 0, 0, 1, rsi) <= 0)
         return 0;
      
      if(trend == 1 && rsi[0] > RSIOverbought) return 0;  // 買いで買われすぎ
      if(trend == -1 && rsi[0] < RSIOversold) return 0;   // 売りで売られすぎ
   }
   
   // リトレースメントチェック
   if(RetracementLevel > 0 && IsAtRetracement(trend))
   {
      return trend;
   }
   
   // 通常のエントリー
   return trend;
}

//+------------------------------------------------------------------+
//| リトレースメントレベルをチェックする関数                        |
//+------------------------------------------------------------------+
bool IsAtRetracement(int trend)
{
   int highSize = ArraySize(swingHighs);
   int lowSize = ArraySize(swingLows);
   
   if(highSize < 2 || lowSize < 2)
      return false;
   
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   if(trend == 1)
   {
      // 上昇トレンドでの押し目
      double lastHigh = swingHighs[highSize-1].price;
      double lastLow = swingLows[lowSize-1].price;
      double retraceLevel = lastLow + (lastHigh - lastLow) * RetracementLevel;
      
      return (currentPrice <= retraceLevel * 1.02 && currentPrice >= retraceLevel * 0.98);
   }
   else if(trend == -1)
   {
      // 下降トレンドでの戻り
      double lastHigh = swingHighs[highSize-1].price;
      double lastLow = swingLows[lowSize-1].price;
      double retraceLevel = lastHigh - (lastHigh - lastLow) * RetracementLevel;
      
      return (currentPrice >= retraceLevel * 0.98 && currentPrice <= retraceLevel * 1.02);
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| スイングポイントを検出する関数                                  |
//+------------------------------------------------------------------+
void DetectSwingPoints()
{
   int bars = iBars(_Symbol, PERIOD_CURRENT);
   if(bars < SwingPeriod * 2 + 1)
      return;
   
   // 配列をクリア
   ArrayResize(swingHighs, 0);
   ArrayResize(swingLows, 0);
   
   int lookback = MathMin(300, bars - SwingPeriod);
   
   for(int i = SwingPeriod; i < lookback; i++)
   {
      bool isSwingHigh = true;
      bool isSwingLow = true;
      
      double currentHigh = iHigh(_Symbol, PERIOD_CURRENT, i);
      double currentLow = iLow(_Symbol, PERIOD_CURRENT, i);
      
      // スイング高値のチェック
      for(int j = 1; j <= SwingPeriod; j++)
      {
         if(i-j >= 0 && iHigh(_Symbol, PERIOD_CURRENT, i-j) >= currentHigh)
            isSwingHigh = false;
         if(i+j < bars && iHigh(_Symbol, PERIOD_CURRENT, i+j) >= currentHigh)
            isSwingHigh = false;
      }
      
      // スイング安値のチェック
      for(int j = 1; j <= SwingPeriod; j++)
      {
         if(i-j >= 0 && iLow(_Symbol, PERIOD_CURRENT, i-j) <= currentLow)
            isSwingLow = false;
         if(i+j < bars && iLow(_Symbol, PERIOD_CURRENT, i+j) <= currentLow)
            isSwingLow = false;
      }
      
      // 最小スイングサイズチェック
      if(MinSwingSize > 0)
      {
         double avgPrice = (currentHigh + currentLow) / 2;
         double minSize = MinSwingSize * point * 10;
         
         if(isSwingHigh)
         {
            bool validSize = false;
            for(int k = 1; k <= SwingPeriod; k++)
            {
               if(currentHigh - iLow(_Symbol, PERIOD_CURRENT, i-k) >= minSize ||
                  currentHigh - iLow(_Symbol, PERIOD_CURRENT, i+k) >= minSize)
               {
                  validSize = true;
                  break;
               }
            }
            if(!validSize) isSwingHigh = false;
         }
         
         if(isSwingLow)
         {
            bool validSize = false;
            for(int k = 1; k <= SwingPeriod; k++)
            {
               if(iHigh(_Symbol, PERIOD_CURRENT, i-k) - currentLow >= minSize ||
                  iHigh(_Symbol, PERIOD_CURRENT, i+k) - currentLow >= minSize)
               {
                  validSize = true;
                  break;
               }
            }
            if(!validSize) isSwingLow = false;
         }
      }
      
      // スイングポイントを記録
      if(isSwingHigh)
      {
         SwingPoint swingHigh;
         swingHigh.price = currentHigh;
         swingHigh.time = iTime(_Symbol, PERIOD_CURRENT, i);
         swingHigh.type = 1;
         swingHigh.barIndex = i;
         
         int size = ArraySize(swingHighs);
         ArrayResize(swingHighs, size + 1);
         swingHighs[size] = swingHigh;
      }
      
      if(isSwingLow)
      {
         SwingPoint swingLow;
         swingLow.price = currentLow;
         swingLow.time = iTime(_Symbol, PERIOD_CURRENT, i);
         swingLow.type = -1;
         swingLow.barIndex = i;
         
         int size = ArraySize(swingLows);
         ArrayResize(swingLows, size + 1);
         swingLows[size] = swingLow;
      }
   }
   
   // 時間順にソート
   SortSwingPoints(swingHighs);
   SortSwingPoints(swingLows);
}

//+------------------------------------------------------------------+
//| スイングポイントをソートする関数                                |
//+------------------------------------------------------------------+
void SortSwingPoints(SwingPoint &points[])
{
   int size = ArraySize(points);
   for(int i = 0; i < size - 1; i++)
   {
      for(int j = i + 1; j < size; j++)
      {
         if(points[i].barIndex > points[j].barIndex)
         {
            SwingPoint temp = points[i];
            points[i] = points[j];
            points[j] = temp;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| トレンドを判定する関数                                          |
//+------------------------------------------------------------------+
int DetermineTrend()
{
   int highSize = ArraySize(swingHighs);
   int lowSize = ArraySize(swingLows);
   
   int requiredSwings = RequireThreeSwings ? 3 : 2;
   
   if(highSize < requiredSwings || lowSize < requiredSwings)
      return 0;
   
   bool isUptrend = true;
   bool isDowntrend = true;
   
   // 複数のスイングで確認
   for(int i = 1; i < requiredSwings; i++)
   {
      // 上昇トレンドチェック
      if(swingHighs[highSize-i].price <= swingHighs[highSize-i-1].price ||
         swingLows[lowSize-i].price <= swingLows[lowSize-i-1].price)
      {
         isUptrend = false;
      }
      
      // 下降トレンドチェック
      if(swingLows[lowSize-i].price >= swingLows[lowSize-i-1].price ||
         swingHighs[highSize-i].price >= swingHighs[highSize-i-1].price)
      {
         isDowntrend = false;
      }
   }
   
   if(isUptrend) return 1;
   if(isDowntrend) return -1;
   
   return 0;
}

//+------------------------------------------------------------------+
//| 買い注文を開く関数                                              |
//+------------------------------------------------------------------+
void OpenBuyOrder()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // 動的ストップロス（直近の安値）
   double dynamicSL = GetDynamicStopLoss(1);
   double sl = MathMin(ask - StopLoss * point * 10, dynamicSL);
   sl = NormalizeDouble(sl, digits);
   
   // リスクリワード比を考慮したTP
   double slDistance = ask - sl;
   double tp = NormalizeDouble(ask + slDistance * 2, digits); // 2:1のリスクリワード
   
   MqlTradeRequest request;
   MqlTradeResult result;
   
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = ORDER_TYPE_BUY;
   request.price = ask;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = "Dow Theory Buy";
   
   if(!OrderSend(request, result))
   {
      Print("買い注文エラー: ", result.comment, " Return code=", result.retcode);
   }
   else
   {
      Print("買い注文成功: チケット=", result.order, " SL=", sl, " TP=", tp);
   }
}

//+------------------------------------------------------------------+
//| 売り注文を開く関数                                              |
//+------------------------------------------------------------------+
void OpenSellOrder()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // 動的ストップロス（直近の高値）
   double dynamicSL = GetDynamicStopLoss(-1);
   double sl = MathMax(bid + StopLoss * point * 10, dynamicSL);
   sl = NormalizeDouble(sl, digits);
   
   // リスクリワード比を考慮したTP
   double slDistance = sl - bid;
   double tp = NormalizeDouble(bid - slDistance * 2, digits); // 2:1のリスクリワード
   
   MqlTradeRequest request;
   MqlTradeResult result;
   
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = ORDER_TYPE_SELL;
   request.price = bid;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = "Dow Theory Sell";
   
   if(!OrderSend(request, result))
   {
      Print("売り注文エラー: ", result.comment, " Return code=", result.retcode);
   }
   else
   {
      Print("売り注文成功: チケット=", result.order, " SL=", sl, " TP=", tp);
   }
}

//+------------------------------------------------------------------+
//| 動的ストップロスを取得する関数                                  |
//+------------------------------------------------------------------+
double GetDynamicStopLoss(int direction)
{
   if(direction == 1) // 買いの場合、直近の安値
   {
      int lowSize = ArraySize(swingLows);
      if(lowSize > 0)
      {
         return swingLows[lowSize-1].price - 10 * point;
      }
   }
   else if(direction == -1) // 売りの場合、直近の高値
   {
      int highSize = ArraySize(swingHighs);
      if(highSize > 0)
      {
         return swingHighs[highSize-1].price + 10 * point;
      }
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| ポジション数を取得する関数                                      |
//+------------------------------------------------------------------+
int GetPositionCount()
{
   int count = 0;
   int total = PositionsTotal();
   
   for(int i = 0; i < total; i++)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            count++;
         }
      }
   }
   
   return count;
}

//+------------------------------------------------------------------+
//| ポジション管理関数                                              |
//+------------------------------------------------------------------+
void ManagePositions()
{
   int total = PositionsTotal();
   
   for(int i = total - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentSL = PositionGetDouble(POSITION_SL);
            double currentTP = PositionGetDouble(POSITION_TP);
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            
            if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
            {
               double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
               
               // ブレークイーブン
               if(UseBreakeven && bid - openPrice >= BreakevenPips * point * 10 && 
                  currentSL < openPrice)
               {
                  double newSL = NormalizeDouble(openPrice + BreakevenProfit * point * 10, digits);
                  ModifyPosition(ticket, newSL, currentTP);
               }
               // トレーリングストップ
               else if(UseTrailingStop && bid - openPrice > TrailingStop * point * 10)
               {
                  double newSL = NormalizeDouble(bid - TrailingStop * point * 10, digits);
                  if(newSL > currentSL + TrailingStep * point * 10)
                  {
                     ModifyPosition(ticket, newSL, currentTP);
                  }
               }
            }
            else if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
            {
               double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               
               // ブレークイーブン
               if(UseBreakeven && openPrice - ask >= BreakevenPips * point * 10 && 
                  currentSL > openPrice)
               {
                  double newSL = NormalizeDouble(openPrice - BreakevenProfit * point * 10, digits);
                  ModifyPosition(ticket, newSL, currentTP);
               }
               // トレーリングストップ
               else if(UseTrailingStop && openPrice - ask > TrailingStop * point * 10)
               {
                  double newSL = NormalizeDouble(ask + TrailingStop * point * 10, digits);
                  if(newSL < currentSL - TrailingStep * point * 10)
                  {
                     ModifyPosition(ticket, newSL, currentTP);
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| ポジションを修正する関数                                        |
//+------------------------------------------------------------------+
void ModifyPosition(ulong ticket, double sl, double tp)
{
   MqlTradeRequest request;
   MqlTradeResult result;
   
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.sl = sl;
   request.tp = tp;
   
   if(!OrderSend(request, result))
   {
      if(ShowDebugInfo)
         Print("ポジション修正エラー: ", result.comment);
   }
}

//+------------------------------------------------------------------+
//| 取引時間かどうかをチェックする関数                              |
//+------------------------------------------------------------------+
bool IsTradeTime()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   if(StartHour < EndHour)
   {
      return (dt.hour >= StartHour && dt.hour < EndHour);
   }
   else // 日をまたぐ場合
   {
      return (dt.hour >= StartHour || dt.hour < EndHour);
   }
}