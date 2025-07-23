//+------------------------------------------------------------------+
//|          EMA_Touch_Simple_EA_v9  – First‑Touch Only              |
//| 2025‑07‑15  (auto‑generated)                                      |
//+------------------------------------------------------------------+
#property version   "9.00"
#property strict

#include <Trade/Trade.mqh>

//─ input
input int  EMA_Fast=20, EMA_Mid=50, EMA_Slow=70;
input double PO_MinDiff=1.0;       // ($)
input bool  ShowLines=true;

input int   Stoch_K=15, Stoch_D=3, Stoch_Slow=9;
input double ExitStochBuy=80, ExitStochSell=20;

input double RiskPercent=1.0;
input double FixedSL_Pips=200;
input int    Magic=909090;

CTrade trade;
//─ handles
int hFast,hMid,hSlow,hStoch;

//─ PO / first‑touch flags
bool POUp=false,PODown=false,POPrev=false;
bool firstTouchUsed=false;

//─ Init
int OnInit()
{
   hFast=iMA(_Symbol,_Period,EMA_Fast,0,MODE_EMA,PRICE_CLOSE);
   hMid =iMA(_Symbol,_Period,EMA_Mid ,0,MODE_EMA,PRICE_CLOSE);
   hSlow=iMA(_Symbol,_Period,EMA_Slow,0,MODE_EMA,PRICE_CLOSE);
   hStoch=iStochastic(_Symbol,_Period,Stoch_K,Stoch_D,Stoch_Slow,MODE_SMA,STO_LOWHIGH);
   return(INIT_SUCCEEDED);
}

//─ helpers
double GetBuf(int h,int shift){ double b[]; ArraySetAsSeries(b,true); CopyBuffer(h,0,shift,1,b); return b[0]; }
double Pip2(double pips){return pips*_Point;}
double CalcLot(double slDist)
{
   double bal=AccountInfoDouble(ACCOUNT_BALANCE);
   double risk=bal*RiskPercent/100.0;
   double tv=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   double ts=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
   double lot=risk/(slDist/ts*tv);
   double step=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   double minL=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   return NormalizeDouble(MathMax(minL,MathFloor(lot/step)*step),2);
}
void VLine(datetime t,color c){ if(!ShowLines) return; string n="L"+IntegerToString(t); if(ObjectFind(0,n)==-1){ObjectCreate(0,n,OBJ_VLINE,0,t,0); ObjectSetInteger(0,n,OBJPROP_COLOR,c);} }

//─ PO update
bool UpdatePO()
{
   double f=GetBuf(hFast,1), m=GetBuf(hMid,1), s=GetBuf(hSlow,1);
   double d1=MathAbs(f-m), d2=MathAbs(m-s);
   POUp   =(f>m && m>s && d1>=PO_MinDiff && d2>=PO_MinDiff);
   PODown =(f<m && m<s && d1>=PO_MinDiff && d2>=PO_MinDiff);
   bool now=(POUp||PODown);
   bool just=( !POPrev && now);
   POPrev=now;
   return just;
}

//─ first‑touch condition
bool FirstTouch(bool buy)
{
   int s=1, prev=2;
   double emaFast_s=GetBuf(hFast,s);
   double emaFast_p=GetBuf(hFast,prev);
   double low_s=iLow(_Symbol,_Period,s);
   double low_p=iLow(_Symbol,_Period,prev);
   double high_s=iHigh(_Symbol,_Period,s);
   double high_p=iHigh(_Symbol,_Period,prev);
   double close_s=iClose(_Symbol,_Period,s);

   if(buy)
      return (low_p>emaFast_p && low_s<emaFast_s && close_s>emaFast_s);
   else
      return (high_p<emaFast_p && high_s>emaFast_s && close_s<emaFast_s);
}

//─ entry
void Buy()
{
   double ask=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double sl=ask-Pip2(FixedSL_Pips);
   double lot=CalcLot(ask-sl);
   if(trade.Buy(lot,_Symbol,ask,sl,0,"FTBuy")){}
}
void Sell()
{
   double bid=SymbolInfoDouble(_Symbol,SYMBOL_BID);
   double sl=bid+Pip2(FixedSL_Pips);
   double lot=CalcLot(sl-bid);
   if(trade.Sell(lot,_Symbol,bid,sl,0,"FTSell")){}
}

//─ tick
void OnTick()
{
   static datetime lastBar=0;
   datetime bt=iTime(_Symbol,_Period,0);
   if(bt==lastBar) return; lastBar=bt;

   bool poJust=UpdatePO();
   if(poJust){ firstTouchUsed=false; VLine(iTime(_Symbol,_Period,1),clrYellow); }

   // exit
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      if(!PositionSelectByTicket(PositionGetTicket(i))) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=Magic) continue;
      ENUM_POSITION_TYPE t=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double k=GetBuf(hStoch,0);
      if( (t==POSITION_TYPE_BUY && k>=ExitStochBuy) ||
          (t==POSITION_TYPE_SELL&& k<=ExitStochSell))
         trade.PositionClose(PositionGetTicket(i));
   }

   if(firstTouchUsed || PositionsTotal()>0) return;

   if(POUp && FirstTouch(true)){ Buy(); firstTouchUsed=true; VLine(bt,clrAqua); }
   else if(PODown && FirstTouch(false)){ Sell(); firstTouchUsed=true; VLine(bt,clrAqua); }
}
