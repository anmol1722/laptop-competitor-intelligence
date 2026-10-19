"""Natural-language chart tools over the catalogue; no generated code or invented values.

Chart planning reuses the RAG query parser. Aggregation runs on the full filtered
catalogue, independently of hosted generation, so chart requests need no model quota.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

import pricing
import sentiment

CHART_VERSION = 1
_CHART = re.compile(r"\b(charts?|graphs?|plots?|visuali[sz]e|visuali[sz]ation|histogram|scatter(?:plot)?|pie|bar chart)\b", re.I)
_PRICE = re.compile(r"\b(pric\w*|cost\w*|expensive|cheap\w*|budget|keemat)\b", re.I)
_SENTIMENT = re.compile(r"\b(sentiment|positive|negative|complaints?|praise|opinions?|satisfaction)\b", re.I)
_COUNT = re.compile(r"\b(counts?|number|listings?|catalogue|catalog|share|popular|popularity)\b", re.I)
_TIME = re.compile(r"\b(over time|price history|trends?|monthly|yearly|by year|by month|forecasts?|predictions?|202[4-9]|203\d|live prices?|current prices?)\b", re.I)


def wants_chart(question: str) -> bool:
    """Explicit plots, plus straightforward numeric brand/segment comparisons."""
    if re.search(r'\b(no charts?|no graphs?|text only|without (?:a )?(?:chart|graph))\b', question, re.I):
        return False
    return bool(_CHART.search(question) or (_PRICE.search(question) and re.search(r'\bdistribution\b', question, re.I)) or (
        re.search(r"\b(compare|comparison|versus|vs)\b", question, re.I)
        and (_PRICE.search(question) or _SENTIMENT.search(question))
        and not re.search(r"\b(no charts?|text only|without (?:a )?(?:chart|graph))\b", question, re.I)))


def _label(value: str) -> str:
    return value.replace('_', ' ').title()


def chart_reply(question: str, products: pd.DataFrame,
                product_sentiment: pd.DataFrame | None = None) -> dict[str, Any] | None:
    if not wants_chart(question):
        return None
    from rag import parse_question  # parsing only; no retriever or model loading

    spec = parse_question(question, list(products['brand'].dropna().unique()))
    q = question.lower()
    # Explicit currency amounts are literal, including values below $15.
    for pattern, attr in ((r'(?:under|below|up to|max(?:imum)?)\s*\$\s*([\d,]+(?:\.\d+)?)\s*(k\b)?', 'price_max'),
                          (r'(?:over|above|at least)\s*\$\s*([\d,]+(?:\.\d+)?)\s*(k\b)?', 'price_min')):
        match = re.search(pattern, q)
        if match:
            setattr(spec, attr, float(match[1].replace(',', '')) * (1000 if match[2] else 1))
    between = re.search(r'between\s*\$\s*([\d,]+(?:\.\d+)?)\s*(?:and|to|-)\s*\$?\s*([\d,]+(?:\.\d+)?)', q)
    if between:
        spec.price_min, spec.price_max = sorted(float(v.replace(',', '')) for v in between.groups())
    notes = ['Amazon Reviews 2023 research snapshot; prices are historical listed USD prices.']
    scope: list[str] = []
    data = products
    filters: dict[str, Any] = {}

    def restrict(key: str, value: Any, mask: pd.Series, description: str) -> None:
        nonlocal data
        data = data.loc[mask.reindex(data.index, fill_value=False)]
        filters[key] = value
        scope.append(description)

    if spec.brands:
        restrict('brands', spec.brands, products.brand.isin(spec.brands), ', '.join(spec.brands))
    if spec.segments:
        restrict('segments', spec.segments, products.segment.isin(spec.segments), ', '.join(spec.segments))
    for key, column, value, op, desc in (
        ('price_min', 'price', spec.price_min, 'ge', 'minimum price'),
        ('price_max', 'price', spec.price_max, 'le', 'maximum price'),
        ('ram_min', 'ram_gb', spec.ram_min, 'ge', 'minimum RAM (GB)'),
        ('storage_min', 'storage_gb', spec.storage_min, 'ge', 'minimum storage (GB)'),
    ):
        if value is not None:
            restrict(key, value, getattr(products[column], op)(value), f'{desc}: {value:g}')
    if spec.os_family and len(spec.segments) <= 1:
        restrict('os_family', spec.os_family, products.os_family.eq(spec.os_family), spec.os_family)
    if spec.screen_in is not None:
        restrict('screen_in', spec.screen_in, products.screen_in.sub(spec.screen_in).abs().le(1), f'screen: {spec.screen_in:g} inches ± 1 inch')
    if spec.cpu_tier is not None:
        restrict('cpu_tier', spec.cpu_tier, products.cpu_tier.eq(spec.cpu_tier), f'CPU tier: {spec.cpu_tier:g}')
    if re.search(r'\b(exclude|without|no)\s+(renewed|refurbished)', q):
        restrict('exclude_renewed', True, ~products.is_renewed, 'renewed listings excluded')
    if re.search(r'\b(discrete|dedicated) (gpu|graphics)|\brtx\b|\bgtx\b', q):
        restrict('discrete_gpu', True, products.is_discrete_gpu, 'discrete GPU')
    if spec.asins:
        restrict('products', spec.asins, products.parent_asin.isin(spec.asins), 'selected product IDs')

    # Product names need exact title evidence, rather than silently turning a named
    # model into a brand-wide aggregate. Extract only meaningful model-name tokens.
    model_tokens = []
    stop = set(re.findall(r'\w+', ' '.join(spec.brands).lower())) | set(re.findall(
        r'\w+', 'price prices cost compare comparison median mean average chart graph bar line pie show plot '
        'sentiment positive negative rating review reviews laptop laptops business gaming ultrabook budget '
        'chromebook mainstream under below above between ram gb storage distribution scatter histogram'))
    for phrase in spec.entity_phrases:
        tokens = [t for t in re.findall(r'[a-z0-9]+', phrase.lower()) if t not in stop and len(t) > 1]
        if tokens and any(re.search(r'\d', t) for t in tokens):
            mask = products.title.str.lower().fillna('').apply(lambda title: all(t in title for t in tokens))
            model_tokens.append((phrase, mask))
    if model_tokens and not spec.asins:
        mask = pd.concat([m for _, m in model_tokens], axis=1).any(axis=1)
        restrict('models', [p for p, _ in model_tokens], mask, 'named models (title matches)')

    total = len(data)
    scope_label = '; '.join(scope) or 'Full catalogue'
    charts: list[dict[str, Any]] = []
    failure = None
    chart: dict[str, Any] = {
        'version': CHART_VERSION, 'id': 'chart-1', 'scope': scope_label, 'filters': filters,
        'source': '[S1]', 'n_total': total, 'rows': [], 'notes': notes,
    }
    group = 'brand' if (len(spec.brands) > 1 or re.search(r'\b(by|per|across) brands?\b', q)) else 'segment'
    if re.search(r'\b(by|per|across) segments?\b', q):
        group = 'segment'
    elif len(spec.segments) == 1 and len(spec.brands) < 2:
        group = 'brand'
    if spec.asins or model_tokens:
        group = 'product'
    pie = bool(re.search(r'\b(pie|donut|doughnut)\b', q))
    is_sentiment = bool(_SENTIMENT.search(q) or (spec.aspects and re.search(r'\b(reviews?|feedback)\b', q)))
    negative = bool(re.search(r'\b(negative|complaints?)\b', q))

    if _TIME.search(q):
        failure = ('A time-series, forecast or current-price chart is unavailable: this dataset '
                   'is a research snapshot and has no historical price observations over time.')
    elif re.search(r'\b(cad|eur|inr|gbp|rupees?|euros?|canadian dollars?)\b|[€£₹]', q):
        failure = 'The dataset lists prices in USD. Currency conversion is unavailable; request a USD chart.'
    elif re.search(r'\bby (?:ram|storage|cpu|gpu|operating system|screen size)\b', q):
        failure = 'Chart groupings currently support brands, segments, named products and review aspects. RAM and storage can be used as minimum filters.'
    elif total == 0:
        failure = 'No catalogue listings match these filters. Try a broader brand, segment or price range.'
    elif re.search(r'\b(scatter(?:plot)?|correlation|relationship|versus|vs)\b', q) and 'rating' in q and _PRICE.search(q):
        valid = data.loc[data.price.gt(0) & data.average_rating.between(1, 5)]
        valid = valid.loc[np.isfinite(valid.price)]
        if valid.empty:
            failure = 'No matching listings have both a known positive price and a rating.'
        else:
            sample = valid.sample(n=min(180, len(valid)), random_state=42).sort_values('parent_asin')
            chart.update(type='scatter', title='Listed price vs customer rating', unit='rating / 5',
                         x_label='Listed price (USD)', y_label='Average rating (1–5)', n_valid=len(valid))
            chart['rows'] = [dict(label=str(r.title), asin=str(r.parent_asin), x=float(r.price),
                                 value=float(r.average_rating), n=int(r.rating_number)) for r in sample.itertuples()]
            notes.append(f'Displaying a reproducible sample of {len(sample):,} of {len(valid):,} eligible listings; this does not establish causation.')
    elif re.search(r'\bscatter(?:plot)?\b', q):
        failure = 'Scatter plots currently support listed price versus average rating. Try: Plot price vs rating for gaming laptops.'
    elif re.search(r'\b(histogram|distribution|spread)\b', q) and _PRICE.search(q):
        values = data.loc[data.price.gt(0) & np.isfinite(data.price), 'price'].to_numpy()
        if not len(values):
            failure = 'No matching listings have a known positive price; unknown prices are excluded.'
        else:
            counts, edges = np.histogram(values, bins=min(10, max(1, len(np.unique(values)))))
            chart.update(type='histogram', title='Distribution of listed prices', unit='listings',
                         x_label='Listed price ranges (USD)', y_label='Listing count', n_valid=len(values))
            chart['rows'] = [dict(label=f'${lo:,.2f}–${hi:,.2f}', low=float(lo), high=float(hi),
                                 value=int(count), n=int(count))
                             for count, lo, hi in zip(counts, edges[:-1], edges[1:])]
            notes.append('Bins include their lower edge; only the final bin also includes its upper edge.')
    elif is_sentiment:
        ps = product_sentiment if product_sentiment is not None else sentiment.load_product_sentiment()
        ps = ps.loc[ps.parent_asin.isin(data.parent_asin)].reset_index(drop=True)
        if ps.empty:
            failure = 'No analysed review sentiment is available for the matching listings.'
        else:
            ps = ps.merge(data[['parent_asin', 'brand', 'segment', 'title']], on='parent_asin', how='inner', validate='one_to_one', suffixes=('', '_catalogue'))
            by_aspect = group not in ('product',) and len(spec.brands) < 2 and not re.search(r'\b(by|per|across) (brands?|segments?)\b', q)
            aspects = spec.aspects or list(sentiment.ASPECT_NAMES)
            if negative:
                reviews = sentiment.load_review_sentiment()
                reviews = reviews.loc[reviews.parent_asin.isin(data.parent_asin)]
                for aspect in (aspects if by_aspect else ([aspects[0]] if spec.aspects else [None])):
                    column = f'asp_{aspect}_pol' if aspect else 'sent_polarity'
                    valid = reviews.loc[reviews[column].notna()]
                    share = valid[column].lt(0).groupby(valid.parent_asin, sort=False).mean()
                    ps[f'{aspect}_neg_share' if aspect else 'overall_neg_share'] = ps.parent_asin.map(share)
            def sentiment_row(label: str, frame: pd.DataFrame, aspect: str | None) -> dict[str, Any] | None:
                ncol, pcol = (f'{aspect}_mentions', f'{aspect}_pos_share') if aspect else ('n_reviews_scored', 'overall_pos_share')
                if negative: pcol = f'{aspect}_neg_share' if aspect else 'overall_neg_share'
                valid = frame.loc[frame[ncol].gt(0) & frame[pcol].between(0, 1)]
                n_reviews = int(valid[ncol].sum())
                if not n_reviews:
                    return None
                value = float((valid[ncol].astype(float) * valid[pcol].astype(float)).sum() / n_reviews * 100)
                return dict(label=label, value=round(value, 4), n=n_reviews, n_products=len(valid), reliable=n_reviews >= 5)
            rows = []
            if by_aspect:
                for aspect in aspects:
                    row = sentiment_row(_label(aspect), ps, aspect)
                    if row: rows.append(row)
            else:
                column = 'title' if group == 'product' else group
                for label, frame in ps.groupby(column, observed=True):
                    row = sentiment_row(str(label), frame, aspects[0] if spec.aspects else None)
                    if row: rows.append(row)
                if len(spec.aspects) > 1:
                    notes.append('The comparison uses the first requested aspect: '+_label(spec.aspects[0])+'.')
            polarity = 'Negative' if negative else 'Positive'
            chart.update(type='bar', title=(polarity+' sentiment by aspect' if by_aspect else
                         (polarity+' '+_label(spec.aspects[0])+' sentiment' if spec.aspects else polarity+' review sentiment')+' by '+group),
                         unit='%', polarity=polarity.lower(), x_label=polarity+' share of analysed reviews (%)', y_label='Aspect' if by_aspect else _label(group),
                         n_valid=len(ps), sample_label='analysed reviews mentioning the aspect' if (by_aspect or spec.aspects) else 'analysed reviews')
            chart['rows'] = sorted(rows, key=lambda r: -r['n'])[:12]
            notes.append('Shares are weighted by review counts, not an average of product percentages. A review can mention multiple aspects. Automated sentiment is imperfect; percentages are not satisfaction ratings.')
            if not rows: failure = 'The requested aspects have no analysed review mentions in the matching listings.'
    elif (_COUNT.search(q) or pie) and not _PRICE.search(q):
        if group == 'product':
            failure = 'For listing counts, request a grouping by brand or segment.'
        else:
            reviews_count = bool(re.search(r'\breviews?\b', q))
            counts = (data.groupby(group, observed=True).n_reviews.sum().sort_values(ascending=False)
                      if reviews_count else data[group].value_counts())
            counts = counts.loc[counts.gt(0)]
            rows = [dict(label=str(label), value=int(value), n=int(value)) for label, value in counts.head(10).items()]
            remaining = int(counts.iloc[10:].sum())
            if remaining: rows.append(dict(label='Other '+group+'s', value=remaining, n=remaining))
            chart.update(type='donut' if pie else 'bar', title=('Retained reviews' if reviews_count else 'Catalogue listings')+' by '+group,
                         unit='reviews' if reviews_count else 'listings', x_label='Retained review count' if reviews_count else 'Listing count',
                         y_label=_label(group), n_valid=total, rows=rows)
            notes.append('Counts describe this dataset, not sales or real-world market share. Other groups are combined when there are more than 10.')
            if reviews_count: notes.append('Review counts refer to retained dataset reviews, not the Amazon rating counts.')
            if not rows: failure = 'No retained reviews are available for the selected groups.'
    elif _PRICE.search(q):
        stat = 'mean' if re.search(r'\b(mean|average)\b', q) else 'median'
        rows = []
        column = 'title' if group == 'product' else group
        for label, frame in data.groupby(column, observed=True):
            values = frame.loc[frame.price.gt(0) & np.isfinite(frame.price), 'price']
            if len(values):
                rows.append(dict(label=str(label), value=round(float(getattr(values, stat)()), 2),
                                 n=len(values), total=len(frame), coverage=len(values)/len(frame), reliable=len(values)>=pricing.MIN_RELIABLE_N))
        if not rows:
            failure = 'No matching listings have a known positive price; missing prices cannot be plotted as zero.'
        else:
            chart.update(type='bar', title=f'{stat.title()} listed price by {group}', unit='USD',
                         x_label=f'{stat.title()} listed price (USD)', y_label=_label(group),
                         n_valid=sum(r['n'] for r in rows), rows=sorted(rows, key=lambda r:-r['value'])[:12])
            notes.append('Only known positive prices are included. Each row shows priced / matching listings; fewer than 5 priced listings are marked as a small sample.')
            if len(rows)>12: notes.append('Showing the 12 groups with the highest '+stat+' listed price; this is not a ranking of sales or popularity.')
    else:
        failure = ('Choose a measurable chart: median price by brand/segment, price distribution, '
                   'price vs rating, catalogue listing counts, or review sentiment by aspect.')
    if re.search(r'\b(line|pie|donut|doughnut)\b', q) and chart.get('type') not in ('donut', None):
        notes.append('A bar/scatter chart is used because these values do not form a time series or parts of a whole.')

    if failure:
        answer = failure+' [S1]'
        evidence_text = '[S1] Chart tool result. '+failure+' Scope: '+scope_label+'.'
        status = 'unavailable'
    else:
        charts.append(chart)
        rows = chart['rows']
        if chart['type']=='scatter':
            summary = f"The chart plots {len(rows):,} of {chart['n_valid']:,} matching listings with known prices and ratings."
        elif chart['type']=='histogram':
            summary = f"The price distribution contains {chart['n_valid']:,} priced listings out of {total:,} matching listings."
        elif chart['unit']=='%':
            leading=max(rows,key=lambda r:r['value'])
            summary=f"Among the displayed groups, {leading['label']} has the highest {chart['polarity']} share at {leading['value']:.1f}%, based on {leading['n']:,} analysed review mentions."
        elif chart['unit']=='USD':
            summary=f"The chart compares {chart['title'].lower()}, using {chart['n_valid']:,} priced listings out of {total:,} matching listings."
            if len(rows) > 1:
                low, high = min(rows, key=lambda r: r['value']), max(rows, key=lambda r: r['value'])
                summary += (f" Among the displayed groups, {low['label']} has the lowest value (${low['value']:,.2f}) "
                            f"and {high['label']} the highest (${high['value']:,.2f}).")
        else:
            summary=(f"The chart shows {sum(r['value'] for r in rows):,} retained dataset reviews across {total:,} matching listings."
                     if chart['unit']=='reviews' else f"The chart shows the distribution of {total:,} matching catalogue listings.")
        answer = summary+' [S1]\nScope: '+scope_label+'. [S1]\n'+notes[0]+' [S1]'
        evidence_text = '[S1] '+summary+' Scope: '+scope_label+'. '+ ' '.join(notes)
        status = 'ready'
    return dict(question_type='chart', answer=answer, charts=charts, chart_status=status,
                response_mode='data_chart', evidence=[dict(marker='[S1]', kind='stat', text=evidence_text,
                meta=dict(source=('products.parquet + product_sentiment.parquet' + (' + review_sentiment.parquet' if negative else '')) if is_sentiment else 'products.parquet',
                          scope=scope_label, filters=filters, n_products=total, chart=charts[0] if charts else None))],
                grounded=True, citation_rate=1.0, citations=[dict(marker='[S1]', kind='stat')],
                unsupported_markers=[], unverified_numbers=[], misattributed_reviews=[], uncited_sentences=[],
                truncated=False, query_spec=spec.to_dict(), retrieval=dict(method='catalogue_chart_tool', n_evidence=1),
                timings={}, prompt_chars=0)
