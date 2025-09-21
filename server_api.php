<?php
header("Content-Type: application/json");
header("Access-Control-Allow-Origin: *");

// Простий LSTM прогноз з мок-даними
function getPrediction($pair = "EURUSD") {
    // Базові ціни для реалістичності
    $basePrices = [
        "EURUSD" => 1.0845, "GBPUSD" => 1.2634, "USDJPY" => 149.87,
        "AUDUSD" => 0.6721, "USDCHF" => 0.8945, "USDCAD" => 1.3567
    ];

    $currentPrice = $basePrices[$pair] ?? 1.0845;

    // Симуляція LSTM прогнозу
    $changePercent = (rand(-200, 200) / 100); // -2% до +2%
    $predictedPrice = $currentPrice * (1 + $changePercent / 100);

    // Рекомендація
    if ($changePercent > 0.5) {
        $action = "BUY";
        $confidence = rand(75, 90);
    } elseif ($changePercent < -0.5) {
        $action = "SELL";
        $confidence = rand(75, 90);
    } else {
        $action = "HOLD";
        $confidence = rand(60, 80);
    }

    return [
        "success" => true,
        "pair" => $pair,
        "current_price" => round($currentPrice, 5),
        "predicted_price" => round($predictedPrice, 5),
        "price_change" => round($predictedPrice - $currentPrice, 5),
        "price_change_percent" => round($changePercent, 2),
        "action" => $action,
        "confidence" => $confidence,
        "timestamp" => date("Y-m-d H:i:s")
    ];
}

// Обробка запитів
$action = $_GET["action"] ?? "health";
$pair = strtoupper($_GET["pair"] ?? "EURUSD");

switch($action) {
    case "predict":
        echo json_encode(getPrediction($pair));
        break;

    case "health":
        echo json_encode([
            "status" => "online",
            "service" => "Simple LSTM Forex Predictor",
            "timestamp" => date("Y-m-d H:i:s")
        ]);
        break;

    default:
        echo json_encode([
            "error" => "Unknown action",
            "available_actions" => ["predict", "health"]
        ]);
}
?>