#!/bin/bash
# Clone every curated repo at the commit this project was tested against.
set -e
V=${CPD_VENDOR:-$HOME/.cache/chart-pattern-detector/vendor}
mkdir -p "$V"
while read -r name repo commit; do
  [ -d "$V/$name" ] || git clone -q https://github.com/$repo "$V/$name"
  git -C "$V/$name" checkout -q "$commit" || echo "warn: $name could not checkout $commit (shallow?), using HEAD"
done <<'LIST'
stock-pattern BennyThadikaran/stock-pattern 85c710d
chart_patterns zeta-zetra/chart_patterns 44f8baa
ta_patterns AdventuresInDataScience/ta_patterns ce791c6
TradingPatternScanner white07S/TradingPatternScanner 57e5396
HarmonicPattern xxwikkixx/HarmonicPattern ff6eacb
pytrendline ednunezg/pytrendline d575f95
ChartScanAI Omar-Karimov/ChartScanAI 58f7120
Stockmarket-pattern-detection foduucom/Stockmarket-pattern-detection 2581739
Kronos shiyu-coder/Kronos 67b630e
chronos-forecasting amazon-science/chronos-forecasting 1224faf
timesfm google-research/timesfm e51928e
big-data-stock-price-forecast nsarang/big-data-stock-price-forecast 5c38bd4
TimeSeries2DBarChartImageCNN omerbsezer/TimeSeries2DBarChartImageCNN ee6cf73
OHLC_Candlestick_Patterns przemyslawbak/OHLC_Candlestick_Patterns 1c24789
LIST
echo "vendor repos in $V"
