import os
filepath = r"C:\Vs code Automation\Trend Alpha 4.0\dashboard.py"

with open(filepath, 'r', encoding='utf-8') as f:
    c = f.read()

# ── FIX 1: Eliminate _CAT_VIS_MAP, reuse _CAT_DISPLAY_MAP for donut chart too ──
# Remove the duplicate _CAT_VIS_MAP dict (lines 5085-5100) and its _vcat function,
# replacing them with a reference to the already-defined _CAT_DISPLAY_MAP (line 5411)
old1 = """        if not _df_ca_vis.empty and \"Core_Weight\" in _df_ca_vis.columns:
            # Normalise category names
            _CAT_VIS_MAP = {
                \"global and inetrnation funds\": \"Global & Intl\",
                \"global and international funds\": \"Global & Intl\",
                \"small cap mutual funds\": \"Small Cap\",
                \"mid cap mutual funds\": \"Mid Cap\",
                \"large cap mutual funds\": \"Large Cap\",
                \"flexi cap mutual funds\": \"Flexi Cap\",
                \"broad market etf or index funds\": \"Broad Market\",
                \"thematic etfs and index funds\": \"Thematic\",
                \"thematic etfs and index funds.\": \"Thematic\",
                \"business cycle and special oportunity fund\": \"Biz Cycle / Opp\",
                \"comodities etfs\": \"Commodities\",
                \"sectoral etfs - index funds\": \"Sectoral\",
                \"strategy etfs and  index funds new\": \"Strategy ETF\",
                \"strategy etfs and index funds new\": \"Strategy ETF\",
            }
            def _vcat(raw):
                c = _norm_cat(raw)
                return _CAT_VIS_MAP.get(c, c.title())

            _df_ca_vis[\"_disp_cat\"] = _df_ca_vis[\"Category\"].apply(_vcat)"""

new1 = """        if not _df_ca_vis.empty and \"Core_Weight\" in _df_ca_vis.columns:
            # Reuse category display map (defined later, but _norm_cat + lookup pattern works)
            def _vcat_display(raw):
                cleaned = str(raw).replace(\".xlsx\", \"\").replace(\"..\", \".\").replace(\"_\", \" \").strip().lower()
                _CMAP = {
                    \"global and inetrnation funds\": \"Global & International Funds\",
                    \"global and international funds\": \"Global & International Funds\",
                    \"small cap mutual funds\": \"Small Cap Mutual Funds\",
                    \"mid cap mutual funds\": \"Mid Cap Mutual Funds\",
                    \"large cap mutual funds\": \"Large Cap Mutual Funds\",
                    \"flexi cap mutual funds\": \"Flexi Cap Mutual Funds\",
                    \"broad market etf or index funds\": \"Broad Market ETF / Index Funds\",
                    \"thematic etfs and index funds\": \"Thematic ETFs & Index Funds\",
                    \"thematic etfs and index funds.\": \"Thematic ETFs & Index Funds\",
                    \"business cycle and special oportunity fund\": \"Business Cycle & Special Opportunities\",
                    \"comodities etfs\": \"Commodities ETFs\",
                    \"sectoral etfs - index funds\": \"Sectoral ETFs & Index Funds\",
                    \"strategy etfs and  index funds new\": \"Strategy ETFs & Index Funds\",
                    \"strategy etfs and index funds new\": \"Strategy ETFs & Index Funds\",
                }
                return _CMAP.get(cleaned, cleaned.title())

            _df_ca_vis[\"_disp_cat\"] = _df_ca_vis[\"Category\"].apply(_vcat_display)"""

assert old1 in c, 'fix1 not found'
c = c.replace(old1, new1)

# ── FIX 2: Replace silent except:pass with info box ──
old2 = """    except Exception as _e_vis:
        pass  # Visualization is non-critical; fail silently

"""
new2 = """    except Exception as _e_vis:
        st.info(f\"📊 Portfolio composition chart unavailable — data incomplete ({_e_vis}). Run pipeline for latest allocations.\")

"""
assert old2 in c, 'fix2 not found'
c = c.replace(old2, new2)

# ── FIX 3: Replace redundant get_global_mf_name_map() with cached _mf_name_map_global ──
old3 = """            st.markdown('##### <EFBFBD><EFBFBD> Unified Core Ranking (All Categories)')
            # BUG FIX 2: Removed redundant raw Symbol column; merged it as secondary label under Fund Name
            _unified_html = ['<div style=\"overflow-x:auto; margin-bottom:25px;\"><table class=\"elimination-table\"><thead><tr>']"""

new3 = """            st.markdown('##### <EFBFBD><EFBFBD> Unified Core Ranking (All Categories)')
            # BUG FIX 2: Removed redundant raw Symbol column; merged it as secondary label under Fund Name
            # Use already-loaded _mf_name_map_global (line 4882) instead of re-calling get_global_mf_name_map()
            _global_mf_n = _mf_name_map_global
            _unified_html = ['<div style=\"overflow-x:auto; margin-bottom:25px;\"><table class=\"elimination-table\"><thead><tr>']"""

assert old3 in c, 'fix3 not found'
c = c.replace(old3, new3)

# Remove the now-redundant line that sets _global_mf_n = get_global_mf_name_map() on the next line
old3b = """            _global_mf_n = get_global_mf_name_map()
            st.markdown('##### <EFBFBD><EFBFBD> Unified Core Ranking (All Categories)')"""
# This should already be fixed by the above replacement, but let me check
assert '            _global_mf_n = get_global_mf_name_map()' in c, 'fix3b - line exists still'
c = c.replace('            _global_mf_n = get_global_mf_name_map()\n', '')

# ── FIX 4: Wrap category analysis table in try/except ──
old4 = """            st.markdown(\"##### <CD> Category-Based Analysis\")\n\n            if \"Category\" in df_core_t.columns:"""

new4 = """            st.markdown(\"##### <CD> Category-Based Analysis\")\n\n            try:\n                if \"Category\" in df_core_t.columns:"""

assert old4 in c, 'fix4 not found'
c = c.replace(old4, new4)

# Add closing except for the try block - find the end of the category table
old4b = """            st.markdown(\"\".join(_core_html_t), unsafe_allow_html=True)
    else:
        st.info(\"Core Portfolio Universe file not found for this date. Run the pipeline to generate L1_Core_Universe.csv.\")"""

new4b = """            st.markdown(\"\".join(_core_html_t), unsafe_allow_html=True)
            except Exception as _e_cat:
                st.warning(f\"⚠️ Category analysis table could not be rendered: {_e_cat}\")
    else:
        st.info(\"Core Portfolio Universe file not found for this date. Run the pipeline to generate L1_Core_Universe.csv.\")"""

assert old4b in c, 'fix4b not found'
c = c.replace(old4b, new4b)

# ── FIX 5: Add named constants for selection weights ──
old5 = """                        # Composite Momentum Score
                        _comp_score = (0.20 * _r15d) + (0.30 * _r1m) + (0.25 * _r2m) + (0.15 * _r3m) + (0.10 * _r6m)"""

new5 = """                        # Composite Momentum Score (time-decay weighted)
                        _W_15D = 0.20; _W_1M = 0.30; _W_2M = 0.25; _W_3M = 0.15; _W_6M = 0.10
                        _comp_score = (_W_15D * _r15d) + (_W_1M * _r1m) + (_W_2M * _r2m) + (_W_3M * _r3m) + (_W_6M * _r6m)"""

assert old5 in c, 'fix5 not found'
c = c.replace(old5, new5)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(c)

print('All 5 Core tab fixes applied OK')
