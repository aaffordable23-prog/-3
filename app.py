
import os
import json
import math
from copy import deepcopy

import requests
import folium
import streamlit as st
from openai import OpenAI
from streamlit_folium import st_folium
from folium.plugins import MiniMap

st.set_page_config(page_title="AI 이동약자 경로 추천", page_icon="🗺️", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      .block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 2rem; }
      .hero {
        background: white; border: 1px solid #e8edf3; border-radius: 24px; padding: 28px;
        box-shadow: 0 6px 22px rgba(17,24,39,0.05); margin-bottom: 18px;
      }
      .title { font-size: 2.1rem; font-weight: 800; color: #111827; margin-bottom: 0.35rem; }
      .subtitle { font-size: 1rem; color: #6b7280; line-height: 1.6; }
      .card {
        background: white; border: 1px solid #e8edf3; border-radius: 22px; padding: 20px;
        box-shadow: 0 6px 22px rgba(17,24,39,0.05); height: 100%;
      }
      .result {
        background: white; border: 1px solid #e8edf3; border-radius: 18px; padding: 16px;
        box-shadow: 0 6px 22px rgba(17,24,39,0.05);
      }
      .muted { color: #6b7280; font-size: 0.95rem; }
      .small { color: #6b7280; font-size: 0.9rem; }
      .stButton > button { border-radius: 14px; padding: 0.7rem 1rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip() if hasattr(st, "secrets") else os.getenv("OPENAI_API_KEY", "")
ORS_API_KEY = st.secrets.get("ORS_API_KEY", os.getenv("ORS_API_KEY", "")).strip() if hasattr(st, "secrets") else os.getenv("ORS_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY가 없습니다. 결론 생성은 제한됩니다.")
if not ORS_API_KEY:
    st.info("ORS_API_KEY가 없습니다. 지도는 표시되지만 실제 경로 계산은 제한될 수 있습니다.")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSM_USER_AGENT = "AI-Navigation-Project/1.0"

DEFAULT_STATE = {
    "origin": "",
    "destination": "",
    "priority": "",
    "mobility": "",
    "constraints": [],
    "notes": [],
    "summary": "",
    "raw_transcript": "",
    "route_compare": {},
    "route_conclusion": "",
    "origin_osm": {},
    "destination_osm": {},
}

if "user_state" not in st.session_state:
    st.session_state.user_state = deepcopy(DEFAULT_STATE)
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

def safe_openai():
    return client is not None

def geocode_place(place_text: str):
    if not place_text:
        return None
    headers = {"User-Agent": OSM_USER_AGENT}
    params = {"q": place_text, "format": "jsonv2", "limit": 1, "countrycodes": "kr"}
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        items = r.json()
        if not items:
            return None
        item = items[0]
        return {"display_name": item["display_name"], "lat": float(item["lat"]), "lon": float(item["lon"])}
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def build_summary_text(state: dict) -> str:
    parts = []
    for key, label in [
        ("origin", "출발지"),
        ("destination", "도착지"),
        ("priority", "우선순위"),
        ("mobility", "이동 형태"),
    ]:
        if state.get(key):
            parts.append(f"{label}: {state[key]}")
    if state.get("constraints"):
        parts.append("조건: " + ", ".join(state["constraints"]))
    if state.get("notes"):
        parts.append("메모: " + "; ".join(state["notes"]))
    if state.get("raw_transcript"):
        parts.append("음성 요약: " + state["raw_transcript"])
    return "\n".join(parts).strip()

def merge_state(base, new_data):
    merged = deepcopy(base)
    for key, value in new_data.items():
        if value in [None, "", []]:
            continue
        if key in {"constraints", "notes"}:
            if not isinstance(value, list):
                value = [value]
            merged.setdefault(key, [])
            for item in value:
                if item not in merged[key]:
                    merged[key].append(item)
        else:
            merged[key] = value
    return merged

def transcribe_audio_file(audio_value):
    if not safe_openai():
        return ""
    try:
        audio_bytes = audio_value.getvalue() if hasattr(audio_value, "getvalue") else audio_value.read()
        from io import BytesIO
        bio = BytesIO(audio_bytes)
        bio.name = getattr(audio_value, "name", "audio.webm")
        result = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=bio)
        return result.text
    except Exception as e:
        st.error(f"음성 인식 실패: {e}")
        return ""

def fix_place_with_gpt(text: str) -> str:
    if not text or not safe_openai():
        return text
    prompt = f"아래 음성인식 결과에서 장소명 오타만 자연스럽게 수정해줘. 문장만 출력:\n{text}"
    try:
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
        return res.choices[0].message.content.strip()
    except Exception:
        return text

def query_overpass_accessibility(lat, lon, radius_m=600):
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius_m},{lat},{lon})[highway=steps];
      way(around:{radius_m},{lat},{lon})[highway=steps];
      node(around:{radius_m},{lat},{lon})[ramp];
      way(around:{radius_m},{lat},{lon})[ramp];
      node(around:{radius_m},{lat},{lon})[wheelchair];
      way(around:{radius_m},{lat},{lon})[wheelchair];
      node(around:{radius_m},{lat},{lon})[highway][incline];
      way(around:{radius_m},{lat},{lon})[highway][incline];
      node(around:{radius_m},{lat},{lon})[amenity=elevator];
      way(around:{radius_m},{lat},{lon})[amenity=elevator];
      node(around:{radius_m},{lat},{lon})[surface];
      way(around:{radius_m},{lat},{lon})[surface];
      node(around:{radius_m},{lat},{lon})[smoothness];
      way(around:{radius_m},{lat},{lon})[smoothness];
      node(around:{radius_m},{lat},{lon})[width];
      way(around:{radius_m},{lat},{lon})[width];
      node(around:{radius_m},{lat},{lon})[kerb];
      way(around:{radius_m},{lat},{lon})[kerb];
    );
    out center tags;
    """
    headers = {"User-Agent": OSM_USER_AGENT}
    try:
        r = requests.get(OVERPASS_URL, params={"data": query}, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json().get("elements", [])
        items = []
        for el in data[:60]:
            tags = el.get("tags", {})
            label = tags.get("name") or tags.get("amenity") or tags.get("highway") or tags.get("ramp") or "object"
            short_tags = {k: tags.get(k) for k in ["wheelchair", "ramp", "incline", "surface", "smoothness", "width", "kerb", "highway", "amenity"] if k in tags}
            items.append({"label": label, "tags": short_tags})
        return {"count": len(data), "items": items}
    except Exception:
        return {"count": 0, "items": []}

def request_route(start, end, profile="foot-walking", wheelchair=False, avoid_steps=False):
    if not ORS_API_KEY:
        return None
    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json, application/geo+json",
    }
    options = {}
    if wheelchair:
        options["profile_params"] = {"restrictions": {"wheelchair": True}}
    if avoid_steps:
        options.setdefault("avoid_features", [])
        options["avoid_features"].append("steps")
    body = {
        "coordinates": [[start["lon"], start["lat"]], [end["lon"], end["lat"]]],
        "instructions": True,
    }
    if options:
        body["options"] = options
    try:
        r = requests.post(url, json=body, headers=headers, timeout=60)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def parse_route_summary(route_json):
    if not route_json or not route_json.get("features"):
        return None
    feature = route_json["features"][0]
    props = feature.get("properties", {})
    summary = props.get("summary", {})
    coords = feature.get("geometry", {}).get("coordinates", [])
    if not coords:
        return None
    return {
        "distance_km": round(summary.get("distance", 0) / 1000, 2),
        "duration_min": round(summary.get("duration", 0) / 60, 1),
        "geometry": coords,
    }

def fallback_route(start, end):
    return {
        "distance_km": round(haversine_km(start["lat"], start["lon"], end["lat"], end["lon"]) * 1.2, 2),
        "duration_min": round(haversine_km(start["lat"], start["lon"], end["lat"], end["lon"]) * 15, 1),
        "geometry": [[start["lon"], start["lat"]], [end["lon"], end["lat"]]],
        "fallback": True,
    }

def compare_routes(user_state):
    origin_text = (user_state.get("origin") or "").strip()
    destination_text = (user_state.get("destination") or "").strip()
    mobility = (user_state.get("mobility") or "").strip().lower()
    constraints = set(user_state.get("constraints") or [])

    if not origin_text or not destination_text:
        return None

    origin = geocode_place(origin_text)
    destination = geocode_place(destination_text)
    if not origin or not destination:
        return None

    result = {
        "origin_geocode": origin,
        "destination_geocode": destination,
        "origin_osm": query_overpass_accessibility(origin["lat"], origin["lon"]),
        "destination_osm": query_overpass_accessibility(destination["lat"], destination["lon"]),
    }

    wheelchair_selected = any(k in constraints for k in ["휠체어", "wheelchair"]) or "휠체어" in mobility
    avoid_steps = any(k in constraints for k in ["계단 최소화", "계단 피하기", "stairs"]) or "계단" in mobility
    mixed_mode = "복합" in mobility or mobility == "" or "모두" in mobility

    profiles = []
    if wheelchair_selected:
        profiles.append(("wheelchair", "wheelchair", True))
        profiles.append(("walking", "foot-walking", True))
    elif "자전거" in mobility or "따릉이" in mobility or "bike" in mobility:
        profiles.append(("bike", "cycling-regular", False))
    elif "자동차" in mobility or "차" in mobility or "택시" in mobility:
        profiles.append(("car", "driving-car", False))
    elif "도보" in mobility or "걷" in mobility:
        profiles.append(("walking", "foot-walking", avoid_steps))
    elif mixed_mode:
        profiles.extend([
            ("walking", "foot-walking", avoid_steps),
            ("bike", "cycling-regular", False),
            ("car", "driving-car", False),
        ])

    # Always try walking first as a safe baseline
    if not profiles:
        profiles.extend([
            ("walking", "foot-walking", avoid_steps),
            ("bike", "cycling-regular", False),
            ("car", "driving-car", False),
        ])

    for key, profile, use_wheelchair in profiles:
        route_json = request_route(origin, destination, profile=profile, wheelchair=use_wheelchair, avoid_steps=avoid_steps)
        parsed = parse_route_summary(route_json)
        if parsed:
            result[key] = parsed
        else:
            result[key] = fallback_route(origin, destination)

    return result

def choose_best_route(compare_data):
    options = [(k, v) for k, v in compare_data.items() if k in {"wheelchair", "walking", "bike", "car"} and isinstance(v, dict)]
    if not options:
        return None, None
    return min(options, key=lambda kv: kv[1].get("duration_min", 10**9))

def create_route_map(compare_data):
    origin = compare_data.get("origin_geocode") if compare_data else None
    destination = compare_data.get("destination_geocode") if compare_data else None

    center = [37.5665, 126.9780]
    if origin and destination:
        center = [(origin["lat"] + destination["lat"]) / 2, (origin["lon"] + destination["lon"]) / 2]

    m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")
    MiniMap(toggle_display=True).add_to(m)

    bounds = []

    if origin:
        folium.Marker(
            [origin["lat"], origin["lon"]],
            popup=f"출발지<br>{origin['display_name']}",
            tooltip="출발지",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        bounds.append([origin["lat"], origin["lon"]])

    if destination:
        folium.Marker(
            [destination["lat"], destination["lon"]],
            popup=f"도착지<br>{destination['display_name']}",
            tooltip="도착지",
            icon=folium.Icon(color="red", icon="flag")
        ).add_to(m)
        bounds.append([destination["lat"], destination["lon"]])

    colors = {"wheelchair":"#7C3AED", "walking":"#EF4444", "bike":"#2563EB", "car":"#10B981"}
    labels = {"wheelchair":"휠체어 경로", "walking":"도보 경로", "bike":"자전거 경로", "car":"자동차 경로"}

    for key in ["wheelchair", "walking", "bike", "car"]:
        route = compare_data.get(key) if compare_data else None
        if not route or not route.get("geometry"):
            continue
        coords = []
        for pt in route["geometry"]:
            try:
                lon, lat = pt[0], pt[1]
                coords.append([lat, lon])
                bounds.append([lat, lon])
            except Exception:
                pass
        if len(coords) >= 2:
            folium.PolyLine(coords, color=colors.get(key, "#111827"), weight=7, opacity=0.95, tooltip=labels.get(key, key)).add_to(m)

    if bounds:
        try:
            m.fit_bounds(bounds)
        except Exception:
            pass
    return m

def generate_route_conclusion(user_state, compare_data, best_key, best_route):
    if not safe_openai():
        return (
            f"추천 경로: {best_key or '확인 필요'}\n"
            f"출발지와 도착지의 접근성 정보를 함께 확인해 선택하세요."
        )
    prompt = f"""
너는 이동약자 맞춤 경로 추천 AI다.

아래 조건과 결과를 바탕으로, 사용자에게 실제 서비스처럼 자연스럽고 짧게 결론을 써라.
반드시 아래 항목을 모두 포함해라:
1. 추천 경로
2. 왜 이 경로가 적합한지
3. 계단/경사로/휠체어 관점의 주의점
4. 출발지/도착지 주변 OSM 접근성 단서
5. 최종 한줄 결론

사용자 조건:
{json.dumps(user_state, ensure_ascii=False)}

경로 비교 결과:
{json.dumps(compare_data, ensure_ascii=False)}

선택된 최종 경로:
{json.dumps({"best_key": best_key, "best_route": best_route}, ensure_ascii=False)}
"""
    try:
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"결론 생성 실패: {e}"

st.markdown(
    """
    <div class="hero">
      <div class="title">AI 이동약자 경로 추천</div>
      <div class="subtitle">계단, 경사로, 휠체어, 엘리베이터 조건을 함께 고려해 경로와 주변 접근성 정보를 보여줍니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("입력")
    mode = st.radio("입력 방식", ["텍스트", "마이크"], horizontal=True)

    origin = st.text_input("출발지", placeholder="예: 강남역")
    destination = st.text_input("도착지", placeholder="예: 서울역")
    priority = st.selectbox("우선순위", ["최단 시간", "최소 환승", "최소 도보", "안전한 길", "이동약자 친화"])

    mobility = st.selectbox("이동 형태", ["일반", "도보", "자전거", "따릉이", "버스/지하철", "자동차", "휠체어", "복합"])

    constraint_options = [
        "계단 최소화",
        "경사로 우선",
        "휠체어 접근성",
        "엘리베이터 우선",
        "낮은 연석",
        "평탄한 표면",
        "폭 넓은 길",
        "미끄럼 적음",
        "문턱 적음",
    ]
    constraints = st.multiselect("세부 조건", constraint_options, default=["계단 최소화"] if mobility == "휠체어" else [])

    max_incline = st.slider("허용 최대 경사(참고용)", 0, 20, 6)
    extra_request = st.text_area("추가 요청사항", height=110, placeholder="예: 비 오는 날이라 미끄럽지 않은 길이 좋습니다.")

    transcript = ""
    if mode == "마이크":
        audio_value = st.audio_input("마이크로 녹음", sample_rate=16000)
        if audio_value is not None:
            with st.spinner("음성을 텍스트로 변환하는 중..."):
                transcript = fix_place_with_gpt(transcribe_audio_file(audio_value))
            st.session_state.last_transcript = transcript
            if transcript:
                st.success("음성 인식 결과가 저장되었습니다.")
                st.caption(transcript)

    run = st.button("AI 추천 받기", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("현재 입력 요약")
    preview_state = {
        "origin": origin,
        "destination": destination,
        "priority": priority,
        "mobility": mobility,
        "constraints": constraints,
        "notes": [extra_request] if extra_request.strip() else [],
        "raw_transcript": transcript or st.session_state.last_transcript,
    }
    preview_text = build_summary_text(preview_state)
    st.write(preview_text if preview_text else "아직 입력된 내용이 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

if run:
    input_state = deepcopy(DEFAULT_STATE)
    input_state["origin"] = origin.strip()
    input_state["destination"] = destination.strip()
    input_state["priority"] = priority
    input_state["mobility"] = mobility
    input_state["constraints"] = list(constraints)
    if extra_request.strip():
        input_state["notes"].append(extra_request.strip())
    transcript_to_use = transcript.strip() or st.session_state.last_transcript.strip()
    if transcript_to_use:
        input_state["raw_transcript"] = transcript_to_use
        input_state["notes"].append(transcript_to_use)

    if not input_state["origin"] or not input_state["destination"]:
        st.warning("출발지와 도착지를 입력해주세요.")
        st.stop()

    input_state["summary"] = build_summary_text(input_state)
    st.session_state.user_state = merge_state(DEFAULT_STATE, input_state)
    st.session_state.user_state["summary"] = input_state["summary"]

    st.markdown("### 입력 내용")
    st.info(st.session_state.user_state["summary"] if st.session_state.user_state["summary"] else "입력 내용이 없습니다.")

    with st.spinner("경로와 접근성 정보를 분석하는 중..."):
        compare_data = compare_routes(st.session_state.user_state)
        st.session_state.user_state["route_compare"] = compare_data or {}

    if not compare_data:
        st.error("출발지 또는 도착지를 찾지 못했거나 경로 계산에 실패했습니다.")
        st.stop()

    best_key, best_route = choose_best_route(compare_data)
    conclusion = generate_route_conclusion(st.session_state.user_state, compare_data, best_key, best_route)
    st.session_state.user_state["route_conclusion"] = conclusion

    st.markdown("## 추천 결과")

    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.subheader("추천 요약")
        if best_route:
            st.write(f"추천 경로: {best_key}")
            st.write(f"총 거리: {best_route['distance_km']} km")
            st.write(f"예상 시간: {best_route['duration_min']} 분")
        else:
            st.write("추천 가능한 경로가 없습니다.")
        st.markdown("---")
        st.write("선택 조건")
        st.write(", ".join(constraints) if constraints else "없음")
        st.write(f"최대 경사 기준: {max_incline}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.subheader("경로 지도")
        st_folium(create_route_map(compare_data), width=900, height=560)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### AI 결론")
    st.markdown(f'<div class="result">{conclusion.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    st.markdown("### OSM 접근성 정보")
    o1, o2 = st.columns(2, gap="large")
    with o1:
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.write("출발지 근처")
        origin_osm = compare_data.get("origin_osm", {})
        st.write(f"감지된 객체 수: {origin_osm.get('count', 0)}")
        for item in origin_osm.get("items", [])[:8]:
            st.write(f"- {item['label']} ({item['tags']})")
        st.markdown('</div>', unsafe_allow_html=True)
    with o2:
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.write("도착지 근처")
        destination_osm = compare_data.get("destination_osm", {})
        st.write(f"감지된 객체 수: {destination_osm.get('count', 0)}")
        for item in destination_osm.get("items", [])[:8]:
            st.write(f"- {item['label']} ({item['tags']})")
        st.markdown('</div>', unsafe_allow_html=True)

    if compare_data.get("wheelchair"):
        st.markdown("### 휠체어/접근성 체크")
        w = compare_data["wheelchair"]
        st.write(f"경로 거리: {w['distance_km']} km / 예상 시간: {w['duration_min']} 분")
