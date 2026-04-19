import MetaTrader5 as mt5
mt5.initialize()

# Find all crypto symbols
symbols = mt5.symbols_get()
print("=== CRYPTO SYMBOLS AVAILABLE ===")
for s in symbols:
    name = s.name.upper()
    if any(x in name for x in ["BTC", "ETH", "LTC", "XRP", "SOL", "DOGE", "ADA", "BNB", "AVAX", "DOT", "LINK", "MATIC", "CRYPTO"]):
        print(f"  {s.name:20} spread={s.spread}  min_lot={s.volume_min}  max_lot={s.volume_max}  lot_step={s.volume_step}  contract={s.trade_contract_size}  visible={s.visible}")

print()
# Also check current account
ai = mt5.account_info()
print(f"Account: {ai.login} | Server: {ai.server} | Balance: {ai.balance} | Equity: {ai.equity}")
print(f"Free margin: {ai.margin_free} | Leverage: {ai.leverage}")

mt5.shutdown()
