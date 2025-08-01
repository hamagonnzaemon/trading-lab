//+------------------------------------------------------------------+
//|  KamuPoint1_EA v1.0 (MT5) – カム式①＋Divergence & V‑Shape＋EMAFilter |
//+------------------------------------------------------------------+
#property strict

/*── 基本パラメータ ───────────────────────────────────────────────*/
input double LotSize            = 0.01;
input double TP_Pips            = 50;
input double SL_Pips            = 30;
input int    Magic              = 112233;

/*── 裁量３ポイントロジック ───────────────────────────────────────*/
input double MaxDistEMA         = 3.0;   // ➊ 200EMAから終値の距離（価格単位）上限
input bool   UseDistEMAFilter   = true;  // ➊ Divergence フィルタ ON/OFF
input bool   UseSlopeFilter     = false;  // ❷ H1‑EMA200上＆横ばい/上向き
input bool   UseVShapeFilter    = true;  // ❸ 前足EMA下抜け→現足反転 V‑Shape
input bool   UseTriggerFilter   = true;  // ➋ 現足でブレイク＆勢い
input double BodyATRMin         = 0.5;   // Trigger勢い判定 (body/ATR > BodyATRMin)

/*── インジハンドル ───────────────────────────────────────────────*/
int hEma20, hEma75, hEma200, hAtr14;

/*── CopyBufferで1本だけ読むヘルパー ────────────────────────────*/
double IndiValue(int handle,int shift=0)
{
   double buf[];
   return CopyBuffer(handle,0,shift,1,buf)==1 ? buf[0] : 0.0;
}

/*───────────────────────────────────────────────────────────────────*/
/*  ポイント① フィルタ本体                                             */
/*───────────────────────────────────────────────────────────────────*/
bool IsKamuPoint1()
{
   // (1) H1‑EMA200 上＆横ばい／上向き
   static int hH1Ema200 = iMA(_Symbol,PERIOD_H1,200,0,MODE_EMA,PRICE_CLOSE);
   double h1c  = iClose(_Symbol,PERIOD_H1,0);
   double h1e0 = IndiValue(hH1Ema200,0);
   double h1e5 = IndiValue(hH1Ema200,5);
   bool slopeOK = (h1c >= h1e0) && (h1e0 >= h1e5);
   if(UseSlopeFilter && !slopeOK) return false;

   // 200EMA の現在値
   double ema200  = IndiValue(hEma200,0);
   double close0  = iClose(_Symbol,PERIOD_M1,0);
   double low1    = iLow  (_Symbol,PERIOD_M1,1);

   // (➊) Divergence: EMAからの乖離距離制限
   if(UseDistEMAFilter && MathAbs(close0 - ema200) > MaxDistEMA)
      return false;

   // (❸) V‑Shape：前足が EMA 下抜け ⇨ 現足が反転上抜け
   bool vShapeOK = (low1 < ema200) && (close0 > low1);
   if(UseVShapeFilter && !vShapeOK) return false;

   // (➋) Break & Momentum（Triggerフィルタ）
   double e20   = IndiValue(hEma20,0),
          e75   = IndiValue(hEma75,0);
   double body0 = MathAbs(close0 - iOpen(_Symbol,PERIOD_M1,0));
   double atrv  = IndiValue(hAtr14,0);
   bool trigOK  = (close0 > low1)
                && (e20 > e75 && e75 > ema200)
                && (body0/atrv > BodyATRMin);
   if(UseTriggerFilter && !trigOK) return false;

   // 全て通過
   return true;
}

/*───────────────────────────────────────────────────────────────────*/
/*  OnInit                                                           */
/*───────────────────────────────────────────────────────────────────*/
int OnInit()
{
   hEma20  = iMA(_Symbol,PERIOD_M1, 20,0,MODE_EMA,PRICE_CLOSE);
   hEma75  = iMA(_Symbol,PERIOD_M1, 75,0,MODE_EMA,PRICE_CLOSE);
   hEma200 = iMA(_Symbol,PERIOD_M1,200,0,MODE_EMA,PRICE_CLOSE);
   hAtr14  = iATR(_Symbol,PERIOD_M1,14);
   if(hEma20==INVALID_HANDLE || hEma75==INVALID_HANDLE
   || hEma200==INVALID_HANDLE|| hAtr14==INVALID_HANDLE)
      return INIT_FAILED;

   Print("✅ KamuPoint1_EA v1.0 initialized");
   return INIT_SUCCEEDED;
}

/*───────────────────────────────────────────────────────────────────*/
/*  OnTick                                                           */
/*───────────────────────────────────────────────────────────────────*/
void OnTick()
{
   // １ポジ制限
   if(PositionSelect(_Symbol)) return;

   // フィルタ通過チェック
   if(!IsKamuPoint1()) return;

   //--- 価格取得
   double ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);

   //--- 安全SL/TP算出
   long   minStop = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
   double spread  = (SymbolInfoDouble(_Symbol,SYMBOL_ASK)
                    -SymbolInfoDouble(_Symbol,SYMBOL_BID))/_Point;
   double safe    = MathMax((double)minStop, SL_Pips) + spread + 5;
   double sl      = ask - MathMax(SL_Pips, safe)*_Point;
   double tp      = ask + MathMax(TP_Pips, safe)*_Point;

   //--- 発注リクエスト
   MqlTradeRequest req; MqlTradeResult res;
   ZeroMemory(req); ZeroMemory(res);
   req.action       = TRADE_ACTION_DEAL;
   req.symbol       = _Symbol;
   req.volume       = LotSize;
   req.type         = ORDER_TYPE_BUY;
   req.price        = ask;
   req.sl           = sl;
   req.tp           = tp;
   req.deviation    = 10;
   req.magic        = Magic;
   req.type_filling = ORDER_FILLING_IOC;  // IOC固定

   if(!OrderSend(req,res))
      Print("❌ OrderSend error: ", GetLastError());
   else
      Print("✅ OrderSend ticket=", res.order);
}
