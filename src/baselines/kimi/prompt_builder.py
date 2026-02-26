def build_entrance_prompt(footprint_wkt: str, allow_empty: bool = False) -> str:
    """
    Constructs the prompt for Kimi K2.5 using the building's geometry.
    """
    empty_rule = "0 to 3" if allow_empty else "1 to 3 (do NOT return empty list)"

    return (
        "Return ONLY valid JSON. No reasoning. No explanation. No extra keys.\n"
        f"You MUST return between {empty_rule} entrances.\n"
        "All entrances should lie ON the building footprint boundary if possible.\n"
        "Schema exactly:\n"
        '{"entrances":[{"lat":0.0,"lon":0.0,"confidence":0.0}]}\n'
        "Rules:\n"
        f"- entrances length: {empty_rule}\n"
        "- confidence in [0,1]\n"
        "Input footprint_wkt:\n"
        f"{footprint_wkt}\n"
    )
