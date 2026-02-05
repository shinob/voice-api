"""気象庁の天気予報APIから今日の天気予報を取得するモジュール"""

import requests
from datetime import datetime


JMA_AREA_URL = "https://www.jma.go.jp/bosai/common/const/area.json"
JMA_FORECAST_URL = "https://www.jma.go.jp/bosai/forecast/data/forecast/{office_code}.json"


def get_area_info(area_code: str) -> dict:
    """地域コードから地域情報と親オフィスコードを取得する"""
    response = requests.get(JMA_AREA_URL)
    response.raise_for_status()
    area_data = response.json()

    # class20s（市区町村）から検索
    if area_code in area_data.get("class20s", {}):
        area = area_data["class20s"][area_code]
        area_name = area["name"]
        parent_code = area["parent"]

        # class15s → class10s → offices と親を辿る
        if parent_code in area_data.get("class15s", {}):
            parent_code = area_data["class15s"][parent_code]["parent"]
        if parent_code in area_data.get("class10s", {}):
            class10_info = area_data["class10s"][parent_code]
            office_code = class10_info["parent"]
            return {
                "area_name": area_name,
                "office_code": office_code,
                "class10_code": parent_code,
                "class10_name": class10_info["name"],
            }

    # class10s（地域）から検索
    if area_code in area_data.get("class10s", {}):
        class10_info = area_data["class10s"][area_code]
        return {
            "area_name": class10_info["name"],
            "office_code": class10_info["parent"],
            "class10_code": area_code,
            "class10_name": class10_info["name"],
        }

    # offices（都道府県）から検索
    if area_code in area_data.get("offices", {}):
        office_info = area_data["offices"][area_code]
        return {
            "area_name": office_info["name"],
            "office_code": area_code,
            "class10_code": None,
            "class10_name": None,
        }

    raise ValueError(f"地域コード '{area_code}' が見つかりません")


def get_today_forecast(area_code: str) -> dict:
    """指定した地域コードの今日の天気予報を取得する"""
    area_info = get_area_info(area_code)
    office_code = area_info["office_code"]

    response = requests.get(JMA_FORECAST_URL.format(office_code=office_code))
    response.raise_for_status()
    forecast_data = response.json()

    # 最初の予報データ（今日・明日・明後日）を取得
    current_forecast = forecast_data[0]
    time_series = current_forecast["timeSeries"]

    # 天気予報（timeSeries[0]）から今日の天気を取得
    weather_series = time_series[0]
    today_date = datetime.now().strftime("%Y-%m-%d")

    # 該当地域のデータを探す
    target_area = None
    for area in weather_series["areas"]:
        area_code_in_forecast = area["area"]["code"]
        # class10_codeと一致するか、または最初の地域を使用
        if area_info["class10_code"] and area_code_in_forecast == area_info["class10_code"]:
            target_area = area
            break

    if target_area is None and weather_series["areas"]:
        target_area = weather_series["areas"][0]

    if target_area is None:
        raise ValueError("天気予報データが見つかりません")

    # 今日の天気を取得
    today_weather = target_area["weathers"][0] if target_area.get("weathers") else None
    today_weather_code = target_area["weatherCodes"][0] if target_area.get("weatherCodes") else None

    # 降水確率（timeSeries[1]）を取得
    pops_series = time_series[1] if len(time_series) > 1 else None
    today_pops = None
    if pops_series:
        for area in pops_series["areas"]:
            if area_info["class10_code"] and area["area"]["code"] == area_info["class10_code"]:
                today_pops = area.get("pops", [])
                break
        if today_pops is None and pops_series["areas"]:
            today_pops = pops_series["areas"][0].get("pops", [])

    # 気温（timeSeries[2]）を取得
    temps_series = time_series[2] if len(time_series) > 2 else None
    today_temps = None
    if temps_series:
        for area in temps_series["areas"]:
            temps = area.get("temps", [])
            if temps:
                today_temps = {
                    "min": temps[0] if len(temps) > 0 else None,
                    "max": temps[1] if len(temps) > 1 else None,
                }
                break

    return {
        "area": area_info["area_name"],
        #"date": today_date,
        "weather": today_weather,
        #"weather_code": today_weather_code,
        "pops": max((int(p) for p in today_pops[:4] if p), default=None) if today_pops else None,  # 今日の降水確率の最大値
        "temps": today_temps,
        #"fetched_at": datetime.now().isoformat(),
        #"report_datetime": current_forecast["reportDatetime"],
    }


if __name__ == "__main__":
    import json
    import sys

    area_code = sys.argv[1] if len(sys.argv) > 1 else "3220100"
    result = get_today_forecast(area_code)
    print(json.dumps(result, ensure_ascii=False, indent=2))
