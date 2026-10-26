"""
ISO 3166-1 alpha-3 country code to continent, used to answer region questions about GDACS events.
"""

_CONTINENT_COUNTRIES: dict[str, str] = {
    "Africa": "DZA AGO BEN BWA BFA BDI CPV CMR CAF TCD COM COG COD CIV DJI EGY GNQ ERI SWZ ETH GAB GMB GHA GIN GNB KEN LSO LBR LBY MDG MWI MLI MRT MUS MYT MAR MOZ NAM NER NGA REU RWA SHN STP SEN SYC SLE SOM ZAF SSD SDN TZA TGO TUN UGA ESH ZMB ZWE",
    "Asia": "AFG ARM AZE BHR BGD BTN BRN KHM CHN CYP GEO HKG IND IDN IRN IRQ ISR JPN JOR KAZ KWT KGZ LAO LBN MAC MYS MDV MNG MMR NPL PRK OMN PAK PSE PHL QAT SAU SGP KOR LKA SYR TWN TJK THA TLS TUR TKM ARE UZB VNM YEM",
    "Europe": "ALB AND AUT BLR BEL BIH BGR HRV CZE DNK EST FRO FIN FRA DEU GIB GRC GGY HUN ISL IRL IMN ITA JEY XKX LVA LIE LTU LUX MLT MDA MCO MNE NLD MKD NOR POL PRT ROU RUS SMR SRB SVK SVN ESP SJM SWE CHE UKR GBR VAT ALA",
    "North America": "AIA ATG ABW BHS BRB BLZ BMU BES VGB CAN CYM CRI CUB CUW DMA DOM SLV GRL GRD GLP GTM HTI HND JAM MTQ MEX MSR NIC PAN PRI BLM KNA LCA MAF SPM VCT SXM TTO TCA USA VIR",
    "South America": "ARG BOL BRA CHL COL ECU FLK GUF GUY PRY PER SUR URY VEN",
    "Oceania": "ASM AUS COK FJI PYF GUM KIR MHL FSM NRU NCL NZL NIU NFK MNP PLW PNG PCN WSM SLB TKL TON TUV VUT WLF",
    "Antarctica": "ATA",
}

ISO3_TO_CONTINENT: dict[str, str] = {code: continent for continent, codes in _CONTINENT_COUNTRIES.items() for code in codes.split()}


# Unique continents for the given country codes, in first-seen order.
def continents_for(iso3_codes: list[str]) -> list[str]:
    seen: list[str] = []
    for code in iso3_codes:
        continent = ISO3_TO_CONTINENT.get(code.strip().upper())
        if continent and continent not in seen:
            seen.append(continent)
    return seen


# Coarse fallback for events without a country (e.g. offshore). Only used when no ISO3 code is available.
def continent_from_lat_lon(lat: float, lon: float) -> str | None:
    if -60 <= lat <= 13 and -82 <= lon <= -34:
        return "South America"
    if 7 <= lat <= 84 and -170 <= lon <= -50:
        return "North America"
    if 35 <= lat <= 72 and -25 <= lon <= 45:
        return "Europe"
    if -35 <= lat <= 37 and -18 <= lon <= 52:
        return "Africa"
    if -50 <= lat <= 0 and 110 <= lon <= 180:
        return "Oceania"
    if -10 <= lat <= 80 and 45 <= lon <= 180:
        return "Asia"
    if lat < -60:
        return "Antarctica"
    return None
