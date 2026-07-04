import os

filepath = r"C:\Vs code Automation\Trend Alpha 4.0\dashboard.py"

with open(filepath, 'r', encoding='utf-8') as f:
    c = f.read()

# Remove broken try: line and fix branch
c = c.replace(
    '\n            try:\n                    if "Category"',
    '\n            if "Category"'
)

# Remove extra indentation from lines inside the old try block
# (4 extra spaces on df_core_t lines + _core_html_t + for loops)
c = c.replace(
    '            if "Category" in df_core_t.columns:\n                    df_core_t["Category"]',
    '            if "Category" in df_core_t.columns:\n                df_core_t["Category"]'
)
c = c.replace(
    '                    df_core_t = df_core_t.sort_values(by=["Category", "Rank"], ascending=[True, True])\n                _core_html_t',
    '                df_core_t = df_core_t.sort_values(by=["Category", "Rank"], ascending=[True, True])\n            _core_html_t'
)
# Fix remaining indentation drift in the table body lines
c = c.replace(
    '                for _h_t in ["Rank", "Fund / ETF',
    '            for _h_t in ["Rank", "Fund / ETF'
)
c = c.replace(
    '                    _core_html_t.append(f"<th>{_h_t}</th>")',
    '                _core_html_t.append(f"<th>{_h_t}</th>")'
)
c = c.replace(
    '                _core_html_t.append("</tr></thead><tbody>")',
    '            _core_html_t.append("</tr></thead><tbody>")'
)
c = c.replace(
    '                _cur_cat_t = None',
    '            _cur_cat_t = None'
)
c = c.replace(
    '                for _, _cr_t in df_core_t.iterrows():',
    '            for _, _cr_t in df_core_t.iterrows():'
)
# Fix the for loop body lines (they start with 20 spaces, should be 16)
c = c.replace(
    '                    # BUG FIX 3: use normalize_cat',
    '                # BUG FIX 3: use normalize_cat'
)
c = c.replace(
    '                    _ccat_t = _normalize_cat',
    '                _ccat_t = _normalize_cat'
)
c = c.replace(
    '                    if _ccat_t != _cur_cat_t:',
    '                if _ccat_t != _cur_cat_t:'
)
c = c.replace(
    '                        _cur_cat_t = _ccat_t',
    '                    _cur_cat_t = _ccat_t'
)
c = c.replace(
    '                        _core_html_t.append(f',
    '                    _core_html_t.append(f'
)
# Fix the remaining iterrows body lines (20 spaces → 16)
c = c.replace(
    '                    _crank_t = int',
    '                _crank_t = int'
)
c = c.replace(
    '                    _cname_t = _global_mf_n',
    '                _cname_t = _global_mf_n'
)
c = c.replace(
    '                    _disp_sym_t = f"',
    '                _disp_sym_t = f"'
)
c = c.replace(
    '                    _cweight_t = float',
    '                _cweight_t = float'
)
c = c.replace(
    '                    _cscore_t = float',
    '                _cscore_t = float'
)
c = c.replace(
    '                    _is_trending_t = bool',
    '                _is_trending_t = bool'
)
c = c.replace(
    '                    _is_buy_elig_t = bool',
    '                _is_buy_elig_t = bool'
)
c = c.replace(
    '                    _cdrawdown_t = float',
    '                _cdrawdown_t = float'
)
c = c.replace(
    '                    _cvol_t = float',
    '                _cvol_t = float'
)
c = c.replace(
    '                    _cprice_t = float',
    '                _cprice_t = float'
)
# Fix badge lines
c = c.replace(
    '                    if _is_trending_t and _is_buy_elig_t:',
    '                if _is_trending_t and _is_buy_elig_t:'
)
c = c.replace(
    '                        _rbadge_t = \'',
    '                    _rbadge_t = \''
)
c = c.replace(
    '                        _rstyle_t = "',
    '                    _rstyle_t = "'
)
c = c.replace(
    '                        _score_str_t =',
    '                    _score_str_t ='
)
c = c.replace(
    '                    if _csym_t in _alloc_syms_t:',
    '                if _csym_t in _alloc_syms_t:'
)
c = c.replace(
    '                    _sbadge_t = \'',
    '                _sbadge_t = \''
)
# The elif/else blocks at 20 spaces
c = c.replace(
    '                    elif _is_buy_elig_t:',
    '                elif _is_buy_elig_t:'
)
c = c.replace(
    '                    else:',
    '                else:'
)
# Fix _dd_color_t, etc.
c = c.replace(
    '                    _dd_color_t = "',
    '                _dd_color_t = "'
)
c = c.replace(
    '                    _dd_str_t = f',
    '                _dd_str_t = f'
)
# Fix the final table row append
c = c.replace(
    '                    _core_html_t.append(f',
    '                _core_html_t.append(f'
)
# Fix closing lines
c = c.replace(
    '                _core_html_t.append("</tbody></table></div>")',
    '            _core_html_t.append("</tbody></table></div>")'
)
c = c.replace(
    '                st.markdown("".join(_core_html_t)',
    '            st.markdown("".join(_core_html_t)'
)

# Remove broken except line
c = c.replace(
    '            except Exception as _e_cat:\n                st.warning',
    'except _BROKEN_\n            # Category table wrapped in try/except for robustness\n'
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(c)

print("Indent cleanup applied")
