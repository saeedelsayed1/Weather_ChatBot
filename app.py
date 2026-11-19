import requests
import streamlit as st
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# =====================================================================
# Page
# =====================================================================
st.set_page_config(page_title="Weather ChatBot", page_icon="🤖")
st.title("🤖 Weather Chatbot")
st.caption("Searches the web and checks the current weather for any city.")

# Arabic lines right-to-left, English lines left-to-right (automatically, per line)
st.markdown(
    """
    <style>
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        unicode-bidi: plaintext;
        text-align: start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# =====================================================================
# Egypt governorates -> cities (lat, lon). First city = governorate capital.
# =====================================================================
EGYPT = {
    "القاهرة": {"القاهرة": (30.0444, 31.2357), "مدينة نصر": (30.0561, 31.3301), "مصر الجديدة": (30.0910, 31.3220),
               "المعادي": (29.9602, 31.2569), "حلوان": (29.8500, 31.3342), "التجمع الخامس": (30.0084, 31.4285)},
    "الجيزة": {"الجيزة": (30.0131, 31.2089), "6 أكتوبر": (29.9285, 30.9188), "الشيخ زايد": (30.0420, 30.9760),
              "العياط": (29.6200, 31.2570)},
    "الإسكندرية": {"الإسكندرية": (31.2001, 29.9187), "برج العرب": (30.9180, 29.5450)},
    "القليوبية": {"بنها": (30.4659, 31.1848), "شبرا الخيمة": (30.1286, 31.2422), "قليوب": (30.1790, 31.2060),
                  "القناطر الخيرية": (30.1930, 31.1360)},
    "الشرقية": {"الزقازيق": (30.5877, 31.5020), "بلبيس": (30.4200, 31.5620), "العاشر من رمضان": (30.2960, 31.7430),
               "منيا القمح": (30.5120, 31.3480)},
    "الدقهلية": {"المنصورة": (31.0409, 31.3785), "طلخا": (31.0530, 31.3780), "ميت غمر": (30.7190, 31.2590),
                "دكرنس": (31.0860, 31.5950), "المنزلة": (31.1580, 31.9380)},
    "الغربية": {"طنطا": (30.7865, 31.0004), "المحلة الكبرى": (30.9690, 31.1660), "كفر الزيات": (30.8240, 30.8150),
               "زفتى": (30.7140, 31.2440)},
    "المنوفية": {"شبين الكوم": (30.5580, 31.0100), "منوف": (30.4650, 30.9310), "أشمون": (30.2970, 30.9760),
                "مدينة السادات": (30.3800, 30.5100)},
    "البحيرة": {"دمنهور": (31.0341, 30.4682), "كفر الدوار": (31.1340, 30.1300), "رشيد": (31.4040, 30.4160),
               "إيتاي البارود": (30.8850, 30.6700)},
    "كفر الشيخ": {"كفر الشيخ": (31.1107, 30.9388), "دسوق": (31.1320, 30.6450), "بلطيم": (31.5550, 31.0900)},
    "دمياط": {"دمياط": (31.4165, 31.8133), "رأس البر": (31.5110, 31.8240), "دمياط الجديدة": (31.4430, 31.6900)},
    "بورسعيد": {"بورسعيد": (31.2653, 32.3019), "بورفؤاد": (31.2500, 32.3200)},
    "الإسماعيلية": {"الإسماعيلية": (30.5965, 32.2715), "فايد": (30.3260, 32.2980), "القنطرة": (30.8540, 32.3200)},
    "السويس": {"السويس": (29.9668, 32.5498), "العين السخنة": (29.6000, 32.3300)},
    "الفيوم": {"الفيوم": (29.3084, 30.8428), "سنورس": (29.4080, 30.8670), "إطسا": (29.2370, 30.7890)},
    "بني سويف": {"بني سويف": (29.0661, 31.0994), "الواسطى": (29.3380, 31.2070)},
    "المنيا": {"المنيا": (28.1099, 30.7503), "ملوي": (27.7310, 30.8410), "مغاغة": (28.6480, 30.8420),
              "سمالوط": (28.3120, 30.7100)},
    "أسيوط": {"أسيوط": (27.1809, 31.1837), "ديروط": (27.5560, 30.8090), "منفلوط": (27.3100, 30.9700)},
    "سوهاج": {"سوهاج": (26.5591, 31.6957), "أخميم": (26.5640, 31.7470), "جرجا": (26.3380, 31.8920),
             "طهطا": (26.7690, 31.5020)},
    "قنا": {"قنا": (26.1551, 32.7160), "نجع حمادي": (26.0490, 32.2410), "قوص": (25.9140, 32.7630)},
    "الأقصر": {"الأقصر": (25.6872, 32.6396), "إسنا": (25.2930, 32.5540), "أرمنت": (25.6150, 32.5330)},
    "أسوان": {"أسوان": (24.0889, 32.8998), "كوم أمبو": (24.4760, 32.9460), "إدفو": (24.9780, 32.8730),
             "أبو سمبل": (22.3370, 31.6260)},
    "البحر الأحمر": {"الغردقة": (27.2579, 33.8116), "سفاجا": (26.7290, 33.9360), "القصير": (26.1040, 34.2800),
                    "مرسى علم": (25.0680, 34.8900)},
    "الوادي الجديد": {"الخارجة": (25.4390, 30.5460), "الداخلة": (25.4920, 28.9790), "الفرافرة": (27.0580, 27.9700)},
    "مطروح": {"مرسى مطروح": (31.3543, 27.2373), "العلمين": (30.8300, 28.9500), "الضبعة": (31.0330, 28.4330),
             "سيوة": (29.2030, 25.5190)},
    "شمال سيناء": {"العريش": (31.1316, 33.7984), "الشيخ زويد": (31.2160, 34.1100), "بئر العبد": (31.0180, 33.0110)},
    "جنوب سيناء": {"الطور": (28.2410, 33.6220), "شرم الشيخ": (27.9158, 34.3300), "دهب": (28.5091, 34.5136),
                  "نويبع": (29.0330, 34.6670), "سانت كاترين": (28.5560, 33.9760)},
}

# =====================================================================
# Weather (Open-Meteo, no API key needed)
# =====================================================================
WEATHER_CODES = {  # code: (English, Arabic)
    0: ("clear sky", "سماء صافية"), 1: ("mainly clear", "صافٍ غالبًا"), 2: ("partly cloudy", "غائم جزئيًا"),
    3: ("overcast", "غائم"), 45: ("fog", "شبورة"), 48: ("rime fog", "شبورة كثيفة"),
    51: ("light drizzle", "رذاذ خفيف"), 53: ("moderate drizzle", "رذاذ"), 55: ("dense drizzle", "رذاذ كثيف"),
    61: ("slight rain", "مطر خفيف"), 63: ("moderate rain", "مطر متوسط"), 65: ("heavy rain", "مطر غزير"),
    71: ("slight snow", "ثلج خفيف"), 73: ("moderate snow", "ثلج"), 75: ("heavy snow", "ثلج كثيف"),
    80: ("slight rain showers", "زخات مطر خفيفة"), 81: ("rain showers", "زخات مطر"),
    82: ("violent rain showers", "زخات مطر غزيرة"), 95: ("thunderstorm", "عاصفة رعدية"),
    96: ("thunderstorm with hail", "عاصفة رعدية مع برد"), 99: ("thunderstorm with heavy hail", "عاصفة رعدية مع برد كثيف"),
}


def fetch_weather(lat: float, lon: float) -> dict:
    """Current weather for coordinates."""
    return requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
        },
        timeout=10,
    ).json()["current"]


def find_city(city: str):
    """City name -> Open-Meteo place dict (tries common spellings, e.g. Mansoura -> Al Mansurah)."""
    name = city.split(",")[0].strip()
    base = name
    for prefix in ("El ", "Al ", "el-", "al-"):
        if base.lower().startswith(prefix.lower()):
            base = base[len(prefix):]
    variants = [name, base, f"Al {base}", f"El {base}",
                base.replace("ou", "u"), f"Al {base.replace('ou', 'u')}",
                f"Al {base.replace('ou', 'u')}h"]
    for v in dict.fromkeys(variants):
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": v, "count": 1},
            timeout=10,
        ).json()
        if geo.get("results"):
            return geo["results"][0]
    return None


def weather_text_ar(label: str, w: dict) -> str:
    """Arabic answer for the sidebar button."""
    desc = WEATHER_CODES.get(w["weather_code"], ("unknown", "غير معروف"))[1]
    return (
        f"**الطقس الآن في {label}:**\n\n"
        f"- الحالة: {desc}\n"
        f"- درجة الحرارة: {w['temperature_2m']}°C (الإحساس الفعلي {w['apparent_temperature']}°C)\n"
        f"- الرطوبة: {w['relative_humidity_2m']}%\n"
        f"- سرعة الرياح: {w['wind_speed_10m']} كم/ساعة"
    )


# =====================================================================
# Tools for the agent
# =====================================================================
@tool
def get_weather_data(city: str) -> str:
    """Get the current weather for a city (temperature, feels like, humidity, wind, description).
    The input MUST be the city name in English only, without country, e.g. 'Mansoura', 'Cairo', 'Alexandria'."""
    try:
        place = find_city(city)
        if place is None:
            return (f"No city found for '{city}'. Try ONE more time with another English spelling "
                    f"(e.g. 'Al Mansurah'). If it fails again, use web_search for the weather.")
        w = fetch_weather(place["latitude"], place["longitude"])
    except Exception as e:
        return f"Weather service error: {e}"

    return (
        f"City: {place['name']}, {place.get('country', '')}\n"
        f"Condition: {WEATHER_CODES.get(w['weather_code'], ('unknown',))[0]}\n"
        f"Temperature: {w['temperature_2m']}°C (feels like {w['apparent_temperature']}°C)\n"
        f"Humidity: {w['relative_humidity_2m']}%\n"
        f"Wind speed: {w['wind_speed_10m']} km/h"
    )


_ddg = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(region="us-en", max_results=5))


@tool
def web_search(query: str) -> str:
    """Search the web for general or up-to-date information. Do NOT use this for weather."""
    try:
        return _ddg.invoke(query)
    except Exception as e:
        return (f"Search failed or found nothing ({e}). "
                f"Answer from your own knowledge, or tell the user you couldn't find it.")


tools = [web_search, get_weather_data]


# =====================================================================
# Agent (built once and cached)
# =====================================================================
@st.cache_resource
def build_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=GROQ_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. "
         "Use get_weather_data for any question about weather. "
         "Always pass the city name to get_weather_data in English (e.g. 'المنصورة' -> 'Mansoura'). "
         "Do not call the same tool with the same input more than once. "
         "Use the web_search tool for general or up-to-date information. "
         "Answer in the same language the user writes in."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True,
                         handle_parsing_errors=True, max_iterations=6)


agent_executor = build_agent()

# =====================================================================
# Session state
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================================
# Sidebar: governorate -> optional city -> weather button
# =====================================================================
with st.sidebar:
    st.header("🌤️ الطقس في مصر")

    gov = st.selectbox("اختار المحافظة", list(EGYPT.keys()), index=None, placeholder="اختار محافظة...")

    city = None
    if gov:
        cities = list(EGYPT[gov].keys())
        city = st.selectbox(
            "اختار المدينة (اختياري)",
            cities[1:],
            index=None,
            placeholder=f"عاصمة المحافظة: {cities[0]}",
        )

    if st.button("اعرض الطقس", disabled=gov is None, use_container_width=True):
        chosen = city or list(EGYPT[gov].keys())[0]
        lat, lon = EGYPT[gov][chosen]
        label = chosen if chosen == gov else f"{chosen} ({gov})"

        st.session_state.messages.append({"role": "user", "content": f"الطقس كام في {label}؟"})
        try:
            answer = weather_text_ar(label, fetch_weather(lat, lon))
        except Exception as e:
            answer = f"معرفتش أجيب الطقس دلوقتي: {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()
    if st.button("🗑️ امسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =====================================================================
# Chat
# =====================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # last 10 messages as memory
    history = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in st.session_state.messages[:-1][-10:]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent_executor.invoke({"input": user_input, "chat_history": history})
                answer = result["output"]
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
