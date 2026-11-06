"""Fixed page copy. Kept apart from the layout so tests can hold it to length limits.

The rails sit beside 400 px charts, so their text has word limits; the method
notes share one row, so they have equal budgets. ``tests/test_text.py`` checks both.
"""

from __future__ import annotations

from typing import Final

from .cards import MethodCard
from .rins import EPA_PAGE

REPOSITORY_URL: Final[str] = "https://github.com/rossi-diego/soybean-crush-margin"
LINKEDIN_URL: Final[str] = "https://www.linkedin.com/in/rossi-diego/"
CME_GUIDE_URL: Final[str] = (
    "https://www.cmegroup.com/education/files/soybean-crush-reference-guide.pdf"
)
FARMDOC_URL: Final[str] = (
    "https://farmdocdaily.illinois.edu/wp-content/uploads/2022/02/fdd021622.pdf"
)
AFDC_URL: Final[str] = "https://afdc.energy.gov/fuels/properties"
EQUIVALENCE_URL: Final[str] = (
    "https://www.ecfr.gov/current/title-40/chapter-I/subchapter-C/part-80/subpart-M/section-80.1415"
)

SCOPE: Final[str] = (
    "Gross margins, before plant, energy, freight and basis costs. Grey sessions are rolls or "
    "deliveries, left out of every statistic. Of the biofuel credits, only the D4 RIN is shown."
)

# "What it is", at most 35 words each.
WHAT: Final[dict[str, str]] = {
    "crush": (
        "A crusher buys soybeans and sells the meal and oil they yield. The margin is what is "
        "left per tonne, before running the plant."
    ),
    "share": (
        "A tonne of soybeans yields about 0.73 t of meal and 0.18 t of oil. Oil is the smaller "
        "volume, but biodiesel demand can make it half the value."
    ),
    "gap": (
        "About 0.90 t of soybean oil makes 1 m³ of biodiesel. This compares that oil's cost with "
        "1 m³ of heating oil, a diesel proxy. Feedstock only, not the biodiesel price."
    ),
}

# "Reading the chart", at most 30 words each, by view.
READING_HISTORY: Final[str] = (
    "Coloured line: clean sessions; the dot is the latest. Thin grey: roll or delivery sessions, "
    "left out of every statistic."
)
READING_BY_MONTH: Final[str] = (
    "Bold: this year; hollow dot: month to date. Colours: years you pick. Band: range of "
    "five previous years. Clean sessions fall mostly late in the month."
)
READING_RIN: Final[str] = (
    "Line: oil minus heating oil and D4 RIN value. Below zero, heating oil plus RIN exceed the "
    "oil cost. Dashed: EPA price held. Dotted: before the RIN."
)

# Under the crush chart's risk strip, at most 30 words.
RISK_NOTE: Final[str] = (
    "1-day historical simulation on clean daily changes of the past 5 years, long 1,000 t "
    "board crush. Front-month closes, not a hedge book."
)

METHOD: Final[list[MethodCard]] = [
    MethodCard(
        "How it is calculated",
        (
            (
                "Crush",
                "0.733 × meal + 0.183 × oil − soybeans, in USD per tonne, from the CME board "
                f'crush yields (<a href="{CME_GUIDE_URL}" target="_blank" rel="noopener">CME</a>).',
            ),
            (
                "Feedstock",
                "0.90 t of soybean oil per m³ of biodiesel, against 1 m³ of NYMEX heating oil "
                f'(<a href="{FARMDOC_URL}" target="_blank" rel="noopener">farmdoc daily</a>).',
            ),
            (
                "D4 RIN",
                "1.5 RINs per gallon × 264.17 gallons per m³ × the weekly EPA price "
                f'(<a href="{EQUIVALENCE_URL}" target="_blank" rel="noopener">40 CFR 80.1415</a>).',
            ),
        ),
        "Ranks: 5-year percentile, clean sessions only.",
    ),
    MethodCard(
        "Rolls and delivery",
        (
            (
                "Series",
                "each leg stays in its nearest contract until the last trading day, then jumps "
                "to the next one.",
            ),
            (
                "Excluded",
                "delivery-period and switch-day sessions are drawn grey and left out of every "
                "rank and monthly average: about 4 in 10 crush sessions.",
            ),
            (
                "Pairs",
                "late September has November beans with October products, and the second half of "
                "November January beans with December; both carry a product calendar spread.",
            ),
        ),
        "Flags follow exchange calendars, not holidays.",
    ),
    MethodCard(
        "What it leaves out",
        (
            (
                "Costs",
                "plant running costs, energy, methanol, freight and local basis: these are gross "
                "margins.",
            ),
            (
                "Credits",
                "the BTC (to 2024), 45Z and LCFS credits are not included, and the D4 RIN goes to "
                "whoever blends the fuel, not necessarily the crusher.",
            ),
            (
                "Data",
                "unadjusted front-month closes; heating oil is NY Harbor ULSD since 2013; the "
                "oil and diesel contracts can be a month apart.",
            ),
        ),
        f'<a href="{REPOSITORY_URL}" target="_blank" rel="noopener">Source and known issues</a>',
    ),
]

FULL_METHOD: Final[str] = (
    "**Crush in CME units.** 0.022 × meal (USD per short ton) + 0.11 × oil (cents per lb) − "
    f"soybeans (cents per bushel) / 100, in USD per bushel ([CME guide]({CME_GUIDE_URL})). The "
    "page shows it per tonne of soybeans: 36.74 bushels.\n\n"
    "**Oil per m³.** 0.90 t per m³ is 7.5 lb of oil per gallon, the average of EIA plant data "
    f"for 2017 to 2020 ([farmdoc daily]({FARMDOC_URL})). Per gallon, the gap is the industry's "
    "BOHO spread.\n\n"
    "**Volume, not energy.** The comparison is per m³, as the market quotes it; biodiesel holds "
    f"about 7% less energy than diesel ([AFDC]({AFDC_URL})).\n\n"
    "**D4 RIN prices.** Weekly volume-weighted averages of EMTS trades, read from EPA's public "
    f"RIN data ([EPA]({EPA_PAGE})). EPA dates each week by its Monday and publishes monthly, so "
    "the latest weeks are held at the last price and drawn dashed. A week's price is applied "
    "from its Monday, so it includes trades later that week. Renewable diesel earns 1.7 RINs "
    "per gallon; only biodiesel is modelled.\n\n"
    "**Monthly averages.** A month-year needs 5 clean sessions, the month in progress 3, and "
    "a month of the five-year band needs 3 of those years. In contract months the delivery "
    "flags leave mostly the second half of the month. Ranks count clean sessions, so months "
    "without a delivery period weigh more. Part of the seasonal shape is the switch to "
    "new-crop contracts.\n\n"
    "**Risk.** Historical simulation on a long 1,000 t board crush: 1-day changes between "
    "consecutive clean sessions over 5 years, so a roll jump is never a P&L. Risk by leg is "
    "each leg's covariance with the crush over its variance."
)
