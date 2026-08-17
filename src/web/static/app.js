/* =========================================================================
 * Laptop Competitor Intelligence — single-page front end
 *
 * Vanilla ES2019+. No build step, no framework. Plotly is optional: it is
 * used for exactly one chart (the price-vs-rating scatter) and that chart
 * degrades to its table twin when the CDN is unreachable. Every other chart
 * is plain HTML/CSS so it renders with zero external dependencies.
 *
 * Data honesty rules mirrored from the API:
 *   - a missing price renders as "price not listed", never 0 and never blank;
 *   - every market statistic renders with its n and its coverage;
 *   - a product with no mined sentiment says so instead of showing an
 *     empty/zero chart.
 * ========================================================================= */
(function () {
  "use strict";

  /* ---------------------------------------------------------------- utils */

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  var MISSING_TOKENS = { "": 1, "unknown": 1, "none": 1, "null": 1, "nan": 1, "other": 1, "n/a": 1 };

  function isMissing(v) {
    if (v === null || v === undefined) return true;
    if (typeof v === "number") return !isFinite(v);
    if (typeof v === "string") return MISSING_TOKENS[v.trim().toLowerCase()] === 1;
    return false;
  }

  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  /** Number with thousands separators; "—" when the value is missing. */
  function n(v, digits) {
    if (isMissing(v)) return "—";
    var d = digits === undefined ? 0 : digits;
    return Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  /** Money. Never returns 0 or "" for a missing price — see priceCell(). */
  function money(v, digits) {
    if (isMissing(v)) return null;
    var d = digits === undefined ? 0 : digits;
    return "$" + Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  function pct(v, digits) {
    if (isMissing(v)) return "—";
    var d = digits === undefined ? 0 : digits;
    return (Number(v) * 100).toFixed(d) + "%";
  }

  /** HTML for a price cell, honouring the API's price_display when present. */
  function priceHTML(price, display) {
    if (isMissing(price)) {
      return '<span class="is-missing">' + esc(display || "price not listed") + "</span>";
    }
    return '<span class="num">' + esc(display || money(price, 2)) + "</span>";
  }

  function specVal(v, suffix) {
    if (isMissing(v)) return '<span class="is-missing">not parsed</span>';
    var s = typeof v === "number" ? n(v, Number.isInteger(v) ? 0 : 1) : esc(v);
    if (v === true) s = "yes";
    if (v === false) s = "no";
    return s + (suffix && v !== true && v !== false ? " " + esc(suffix) : "");
  }

  function truncate(s, len) {
    s = String(s || "");
    return s.length > len ? s.slice(0, len - 1) + "…" : s;
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  var toastTimer = null;
  function toast(msg) {
    var el = $("#toast");
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.hidden = true; }, 4200);
  }

  /* ------------------------------------------------------------ api client */

  function ApiError(message, status, body) {
    this.name = "ApiError";
    this.message = message;
    this.status = status;
    this.body = body;
  }
  ApiError.prototype = Object.create(Error.prototype);

  function api(path, options) {
    var opts = options || {};
    return fetch(path, opts).then(function (res) {
      return res.text().then(function (text) {
        var body = null;
        if (text) {
          try { body = JSON.parse(text); } catch (e) {
            throw new ApiError(
              "The server returned a body that is not valid JSON (" + res.status + ").",
              res.status, text.slice(0, 400));
          }
        }
        if (!res.ok) {
          var msg = (body && body.error && body.error.message) || (body && body.detail) || res.statusText;
          if (typeof msg !== "string") msg = JSON.stringify(msg).slice(0, 300);
          throw new ApiError(msg, res.status, body);
        }
        return body;
      });
    }, function (netErr) {
      throw new ApiError("Cannot reach the API (" + netErr.message + "). Is the server running?", 0, null);
    });
  }

  function qs(params) {
    var parts = [];
    Object.keys(params).forEach(function (k) {
      var v = params[k];
      if (v === null || v === undefined || v === "") return;
      parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(v));
    });
    return parts.length ? "?" + parts.join("&") : "";
  }

  /* ----------------------------------------------------- shared UI states */

  function loadingHTML(label) {
    return '<div class="state state--loading"><div class="state__title">' +
      '<span class="spinner"></span> ' + esc(label || "Loading…") + "</div></div>";
  }

  function errorHTML(err, retryId) {
    var status = err && err.status ? " (HTTP " + err.status + ")" : "";
    return '<div class="state state--error">' +
      '<div class="state__title">Could not load this view' + esc(status) + "</div>" +
      '<div class="state__body">' + esc(err && err.message ? err.message : String(err)) + "</div>" +
      (retryId ? '<div style="margin-top:10px"><button type="button" class="btn btn--sm" id="' +
        esc(retryId) + '">Retry</button></div>' : "") +
      "</div>";
  }

  function emptyHTML(title, body) {
    return '<div class="state"><div class="state__title">' + esc(title) + "</div>" +
      '<div class="state__body">' + esc(body || "") + "</div></div>";
  }

  /* ------------------------------------------------------------ app state */

  var STATE = {
    overview: null,
    segments: null,
    brands: null,
    focusAsin: null,
    detail: null,
    search: { params: null, items: [], total: 0, offset: 0 },
    competitors: { guarded: null, unguarded: null, k: 12 },
    reviews: null,
    activeTab: "overview",
    loaded: { overview: false, competitors: false, pricing: false, reviews: false },
    chat: { msgSeq: 0, busy: false, llm: null, pollTimer: null }
  };

  var PLOTLY_OK = function () {
    return !window.__PLOTLY_BLOCKED__ && typeof window.Plotly !== "undefined";
  };

  /* ==================================================================== */
  /* 1. Header: market overview stat strip + health pill                  */
  /* ==================================================================== */

  function statTile(value, label, denom, cls) {
    return '<div class="stat' + (cls ? " " + cls : "") + '">' +
      '<div class="stat__value">' + value + "</div>" +
      '<div class="stat__label">' + esc(label) + "</div>" +
      '<div class="stat__denom">' + denom + "</div></div>";
  }

  function renderStatStrip() {
    var el = $("#statStrip");
    var ov = STATE.overview;
    el.setAttribute("aria-busy", "false");
    if (!ov) {
      el.innerHTML = '<div class="stat stat--error"><div class="stat__value">—</div>' +
        '<div class="stat__label">market overview</div>' +
        '<div class="stat__denom">failed to load; the panels below still work</div></div>';
      return;
    }
    var c = ov.catalogue, p = ov.price, s = ov.sentiment;
    el.innerHTML = [
      statTile('<span class="num">' + n(c.n_products) + "</span>", "laptops",
        "distinct listings in products.parquet"),
      statTile('<span class="num">' + n(c.n_brands) + "</span>", "brands",
        "across " + n(c.n_segments) + " segments"),
      statTile('<span class="num">' + n(s.n_reviews_scored) + "</span>", "reviews analysed",
        "of " + n(s.n_reviews_available) + " retained (" +
        pct(s.n_reviews_scored / (s.n_reviews_available || 1), 1) + ")"),
      statTile('<span class="num">' + pct(p.coverage, 1) + "</span>", "price coverage",
        n(p.n_priced) + " of " + n(p.n_total) + " listings priced"),
      statTile('<span class="num">' + pct(s.coverage, 1) + "</span>", "sentiment coverage",
        n(s.n_products_with_sentiment) + " of " + n(s.n_products) + " products mined")
    ].join("");
  }

  function renderHealthPill(h, err) {
    var pill = $("#healthPill");
    if (err) {
      pill.className = "pill pill--bad";
      pill.innerHTML = '<span class="pill__dot"></span>API unreachable';
      pill.title = err.message;
      return;
    }
    var caches = (h.caches && h.caches.state) || "unknown";
    var ok = h.status === "ok";
    pill.className = "pill " + (ok ? (caches === "warm" ? "pill--ok" : "pill--busy") : "pill--warn");
    pill.innerHTML = '<span class="pill__dot"></span>API ' + esc(h.status) +
      " &middot; caches " + esc(caches);
    pill.title = "products " + n(h.rows.products) + " · reviews " + n(h.rows.reviews) +
      " · product_sentiment " + n(h.rows.product_sentiment) +
      "\nstartup " + (h.startup_load_s === null ? "?" : Number(h.startup_load_s).toFixed(2)) + " s" +
      "\nclick to refresh";
  }

  function loadHealth() {
    return api("/api/health").then(function (h) {
      renderHealthPill(h, null);
      return h;
    }, function (err) {
      renderHealthPill(null, err);
      throw err;
    });
  }

  function loadOverview() {
    return api("/api/market/overview").then(function (ov) {
      STATE.overview = ov;
      renderStatStrip();
      return ov;
    }, function (err) {
      STATE.overview = null;
      renderStatStrip();
      toast("Market overview failed to load: " + err.message);
      throw err;
    });
  }

  /* ==================================================================== */
  /* 2. Left column: search & filters                                     */
  /* ==================================================================== */

  function readFilters() {
    var gpu = $("#fGpu").value;
    return {
      q: $("#fQ").value.trim() || null,
      brand: $("#fBrand").value.trim() || null,
      segment: $("#fSegment").value || null,
      min_price: $("#fMinPrice").value !== "" ? Number($("#fMinPrice").value) : null,
      max_price: $("#fMaxPrice").value !== "" ? Number($("#fMaxPrice").value) : null,
      min_ram: $("#fRam").value !== "" ? Number($("#fRam").value) : null,
      has_discrete_gpu: gpu === "" ? null : gpu,
      is_renewed: $("#fRenewedOut").checked ? "false" : null,
      has_sentiment: $("#fSent").checked ? "true" : null,
      sort: $("#fSort").value,
      limit: 25
    };
  }

  function resultHTML(item) {
    var tags = [];
    if (item.is_renewed) tags.push('<span class="tag tag--renewed">renewed</span>');
    if (item.has_sentiment) tags.push('<span class="tag tag--sent">reviews mined</span>');
    if (item.specs && item.specs.is_discrete_gpu) tags.push('<span class="tag">discrete GPU</span>');
    var ram = isMissing(item.specs && item.specs.ram_gb) ? null : n(item.specs.ram_gb) + " GB";
    var rating = isMissing(item.average_rating) ? "unrated"
      : Number(item.average_rating).toFixed(1) + "★ (" + n(item.rating_number) + ")";

    return '<button type="button" class="result' +
      (item.parent_asin === STATE.focusAsin ? " is-focus" : "") +
      '" data-asin="' + esc(item.parent_asin) + '">' +
      '<div class="result__title">' + esc(item.title) + "</div>" +
      '<div class="result__meta">' +
      '<span class="result__price' + (item.price_available ? "" : " is-missing") + '">' +
      esc(item.price_display) + "</span>" +
      "<span>" + esc(item.brand) + "</span>" +
      "<span>" + esc(item.segment) + "</span>" +
      (ram ? "<span>" + ram + "</span>" : "") +
      "<span>" + rating + "</span>" +
      "</div>" +
      (tags.length ? '<div class="result__tags">' + tags.join("") + "</div>" : "") +
      "</button>";
  }

  function renderSearchResults(append) {
    var box = $("#results");
    var s = STATE.search;
    if (!append) box.innerHTML = "";
    if (!s.items.length) {
      box.innerHTML = emptyHTML("No listings match those filters",
        "Loosen a filter — note that a price filter necessarily drops the ~70% of the " +
        "catalogue with no listed price.");
    } else {
      var html = s.items.slice(append ? box.querySelectorAll(".result").length : 0)
        .map(resultHTML).join("");
      box.insertAdjacentHTML("beforeend", html);
    }
    $("#loadMore").hidden = s.items.length >= s.total || !s.items.length;
  }

  function runSearch(append) {
    var box = $("#results");
    var params = append ? STATE.search.params : readFilters();
    var offset = append ? STATE.search.items.length : 0;
    params.offset = offset;
    STATE.search.params = params;

    if (params.min_price !== null && params.max_price !== null && params.min_price > params.max_price) {
      box.innerHTML = errorHTML({ message: "Minimum price is above the maximum price.", status: 0 });
      $("#searchMeta").textContent = "";
      return;
    }

    box.setAttribute("aria-busy", "true");
    if (!append) box.innerHTML = loadingHTML("Searching the catalogue…");
    $("#loadMore").disabled = true;

    api("/api/products/search" + qs(params)).then(function (data) {
      box.setAttribute("aria-busy", "false");
      $("#loadMore").disabled = false;
      STATE.search.total = data.total;
      STATE.search.items = append ? STATE.search.items.concat(data.items) : data.items;
      if (!append) box.innerHTML = "";
      renderSearchResults(append);

      $("#searchMeta").innerHTML =
        "<span><strong>" + n(data.total) + "</strong> of " + n(data.coverage.catalogue_size) +
        " listings match</span>" +
        "<span>" + pct(data.coverage.price_coverage_in_result, 0) + " priced &middot; " +
        n(data.took_ms, 0) + " ms</span>";

      var notes = (data.notes || []).slice();
      if (data.query.sort !== data.query.effective_sort) {
        // the API already explains this in notes; nothing extra needed
      }
      var nb = $("#searchNotes");
      if (notes.length) {
        nb.hidden = false;
        nb.innerHTML = "<ul>" + notes.map(function (t) { return "<li>" + esc(t) + "</li>"; }).join("") + "</ul>";
      } else {
        nb.hidden = true;
        nb.innerHTML = "";
      }
    }, function (err) {
      box.setAttribute("aria-busy", "false");
      $("#loadMore").disabled = false;
      box.innerHTML = errorHTML(err, "retrySearch");
      var r = $("#retrySearch");
      if (r) r.addEventListener("click", function () { runSearch(false); });
      $("#searchMeta").textContent = "";
    });
  }

  function loadFilterOptions() {
    api("/api/segments").then(function (data) {
      STATE.segments = data;
      var sel = $("#fSegment");
      var rows = (data.rows || []).slice().sort(function (a, b) {
        return b.n_products - a.n_products;
      });
      rows.forEach(function (r) {
        var o = document.createElement("option");
        o.value = r.segment;
        o.textContent = r.segment + " (" + n(r.n_products) + ")";
        sel.appendChild(o);
      });
    }, function () { /* the select simply stays at "any" */ });

    api("/api/brands" + qs({ limit: 300, sort: "n_products", desc: true })).then(function (data) {
      STATE.brands = data;
      var dl = $("#brandList");
      (data.rows || []).forEach(function (r) {
        var o = document.createElement("option");
        o.value = r.brand;
        o.label = r.brand + " — " + n(r.n_products) + " listings";
        dl.appendChild(o);
      });
    }, function () { /* datalist is a convenience only */ });
  }

  /* ==================================================================== */
  /* 3. Focus product + tab plumbing                                      */
  /* ==================================================================== */

  function renderFocusBar() {
    var bar = $("#focusBar");
    var d = STATE.detail;
    if (!STATE.focusAsin) {
      bar.innerHTML = '<div class="focusbar__empty">No focus product selected — search on the ' +
        "left and pick a listing.</div>";
      return;
    }
    if (!d) {
      bar.innerHTML = '<div class="focusbar__empty"><span class="spinner"></span> Loading ' +
        esc(STATE.focusAsin) + "…</div>";
      return;
    }
    var pos = d.price_position || {};
    var seg = pos.vs_segment || {};
    bar.innerHTML = '<div class="focusbar__row">' +
      '<div class="focusbar__title">' + esc(truncate(d.title, 130)) +
      '<div class="muted" style="font-weight:400;font-size:11.5px;margin-top:2px">' +
      esc(d.brand) + " &middot; " + esc(d.segment) + " &middot; " + esc(d.parent_asin) +
      (d.is_renewed ? " &middot; renewed" : "") + "</div></div>" +
      '<div class="focusbar__kv">' +
      '<div><div class="kv__k">price</div><div class="kv__v' +
      (d.price_available ? "" : " is-missing") + '">' + esc(d.price_display) + "</div></div>" +
      '<div><div class="kv__k">rating</div><div class="kv__v">' +
      (isMissing(d.market.average_rating) ? '<span class="is-missing">unrated</span>'
        : Number(d.market.average_rating).toFixed(1) + "★") +
      '</div><div class="muted" style="font-size:10.5px">n=' + n(d.market.rating_number) + "</div></div>" +
      '<div><div class="kv__k">vs segment</div><div class="kv__v">' +
      esc(seg.label || "unknown") + "</div>" +
      '<div class="muted" style="font-size:10.5px">n=' + n(seg.n) + " priced peers</div></div>" +
      '<div><div class="kv__k">reviews mined</div><div class="kv__v">' +
      (d.no_sentiment_data ? '<span class="is-missing">none yet</span>'
        : n(d.reviews.n_reviews_scored)) + "</div></div>" +
      "</div></div>";
  }

  function setFocus(asin) {
    if (!asin) return;
    STATE.focusAsin = asin;
    STATE.detail = null;
    STATE.competitors = { guarded: null, unguarded: null, k: STATE.competitors.k };
    STATE.reviews = null;
    STATE.loaded.overview = STATE.loaded.competitors = STATE.loaded.reviews = false;
    $$(".result").forEach(function (b) {
      b.classList.toggle("is-focus", b.getAttribute("data-asin") === asin);
    });
    renderFocusBar();
    renderTab(STATE.activeTab, true);

    api("/api/products/" + encodeURIComponent(asin)).then(function (d) {
      if (STATE.focusAsin !== asin) return;
      STATE.detail = d;
      renderFocusBar();
      if (STATE.activeTab === "overview") renderOverviewTab();
      if (STATE.activeTab === "pricing") renderPricingTab();
    }, function (err) {
      if (STATE.focusAsin !== asin) return;
      STATE.detail = null;
      $("#focusBar").innerHTML = '<div class="focusbar__empty">Could not load ' + esc(asin) +
        ": " + esc(err.message) + "</div>";
      if (STATE.activeTab === "overview") {
        $("#panel-overview").innerHTML = errorHTML(err, "retryDetail");
        var r = $("#retryDetail");
        if (r) r.addEventListener("click", function () { setFocus(asin); });
      }
    });
  }

  function activateTab(name) {
    STATE.activeTab = name;
    $$(".tab").forEach(function (t) {
      var on = t.getAttribute("data-tab") === name;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    $$(".panel").forEach(function (p) {
      var on = p.id === "panel-" + name;
      p.classList.toggle("is-active", on);
      p.hidden = !on;
    });
    renderTab(name, false);
  }

  function renderTab(name, force) {
    if (name === "overview") renderOverviewTab();
    else if (name === "competitors") renderCompetitorsTab(force);
    else if (name === "pricing") renderPricingTab(force);
    else if (name === "reviews") renderReviewsTab(force);
  }

  var NEED_FOCUS = "Select a product from the search results on the left to populate this view.";

  /* ==================================================================== */
  /* 4. Overview tab — spec card + price position ruler                   */
  /* ==================================================================== */

  function specCardHTML(d) {
    var s = d.specs || {};
    var rows = [
      ["CPU", isMissing(s.cpu_brand) && isMissing(s.cpu_family) ? null :
        [s.cpu_brand, s.cpu_family, s.cpu_tier].filter(function (x) { return !isMissing(x); }).join(" ")],
      ["CPU clock", isMissing(s.cpu_ghz) ? null : n(s.cpu_ghz, 1) + " GHz"],
      ["RAM", isMissing(s.ram_gb) ? null : n(s.ram_gb) + " GB" + (isMissing(s.ram_type) ? "" : " " + s.ram_type)],
      ["Storage", isMissing(s.storage_gb) ? null : n(s.storage_gb) + " GB" + (isMissing(s.storage_type) ? "" : " " + s.storage_type)],
      ["Screen", isMissing(s.screen_in) ? null : n(s.screen_in, 1) + '"' + (s.screen_res ? " · " + s.screen_res : "")],
      ["GPU", isMissing(s.gpu_model) ? (isMissing(s.gpu_brand) ? null : s.gpu_brand + " (integrated)") :
        s.gpu_model + (s.is_discrete_gpu ? " · discrete" : " · integrated")],
      ["OS", isMissing(s.os_family) ? null : s.os_family],
      ["Weight", isMissing(s.weight_lb) ? null : n(s.weight_lb, 1) + " lb"],
      ["Store", isMissing(d.store) ? null : d.store],
      ["Variants", n(d.market.n_variants)]
    ];
    return '<div class="specgrid">' + rows.map(function (r) {
      return "<div><div class=\"spec__k\">" + esc(r[0]) + "</div><div class=\"spec__v" +
        (r[1] === null ? " is-missing" : "") + "\">" + (r[1] === null ? "not parsed" : esc(r[1])) +
        "</div></div>";
    }).join("") + "</div>";
  }

  /**
   * A price-position ruler: the peer distribution p10–p90 with its IQR box and
   * median, and the focus product's own price marked on it. When the product has
   * no price the peer distribution is still drawn and the marker is replaced by
   * an explicit "price not listed" statement.
   */
  function rulerHTML(block, price, groupLabel) {
    if (!block || isMissing(block.median) || block.n === 0) {
      return '<div class="callout callout--warn"><span class="callout__icon">!</span><span>' +
        "No priced peers in " + esc(groupLabel) + " — no position can be computed " +
        "(n=" + n(block ? block.n : 0) + " priced of " + n(block ? block.n_total : 0) + ").</span></div>";
    }
    // The peer percentiles are FACTS about the peer group; the axis is only a
    // drawing range. When this product's price falls outside p10..p90 the axis
    // has to stretch to fit the marker — but the end tags must keep reporting the
    // real p10/p90, never relabel this listing's own price as a peer percentile.
    var p10 = isMissing(block.p10) ? block.min : block.p10;
    var p90 = isMissing(block.p90) ? block.max : block.p90;
    var loName = isMissing(block.p10) ? "min" : "p10";
    var hiName = isMissing(block.p90) ? "max" : "p90";
    var lo = p10, hi = p90;
    var stretched = false;
    if (!isMissing(price)) {
      if (price < lo) { lo = price; stretched = true; }
      if (price > hi) { hi = price; stretched = true; }
    }
    var span = Math.max(hi - lo, 1e-6);
    var at = function (v) { return Math.max(0, Math.min(100, ((v - lo) / span) * 100)); };

    var iqrL = at(isMissing(block.p25) ? block.median : block.p25);
    var iqrR = at(isMissing(block.p75) ? block.median : block.p75);

    var marker = "";
    if (!isMissing(price)) {
      var mp = at(price);
      marker = '<div class="ruler__marker" style="left:' + mp.toFixed(2) + '%"></div>' +
        '<div class="ruler__flag" style="left:' + Math.max(6, Math.min(94, mp)).toFixed(2) + '%">' +
        "this product " + esc(money(price, 2)) + "</div>";
    }

    var pctText = isMissing(block.percentile) ? null :
      Number(block.percentile).toFixed(1) + "th percentile";

    return '<div class="ruler">' +
      '<div class="ruler__track" data-tip="' + esc(
        "p25 " + (money(block.p25, 0) || "n/a") + " · median " + (money(block.median, 0) || "n/a") +
        " · p75 " + (money(block.p75, 0) || "n/a") +
        "\nn=" + n(block.n) + " priced of " + n(block.n_total)) + '">' +
      '<div class="ruler__iqr" style="left:' + iqrL.toFixed(2) + "%;width:" +
      Math.max(0.6, iqrR - iqrL).toFixed(2) + '%"></div>' +
      '<div class="ruler__median" style="left:' + at(block.median).toFixed(2) + '%"></div>' +
      marker + "</div>" +
      '<div class="ruler__tags"><span>' + esc(money(p10, 0)) + " (" + loName + ")</span>" +
      "<span>median " + esc(money(block.median, 0)) + "</span>" +
      "<span>" + esc(money(p90, 0)) + " (" + hiName + ")</span></div>" +
      '<div class="chart__legend" style="margin-top:8px">' +
      '<span class="legend__item"><span class="legend__swatch" style="background:var(--div-mid);border:1px solid var(--axis)"></span>middle 50% of priced peers</span>' +
      '<span class="legend__item"><span class="legend__swatch" style="background:var(--axis)"></span>peer median</span>' +
      (isMissing(price) ? "" :
        '<span class="legend__item"><span class="legend__swatch" style="background:var(--series-2)"></span>this product</span>') +
      "</div>" +
      '<div class="muted" style="margin-top:8px;font-size:11.5px">Based on <strong class="num">' +
      n(block.n) + "</strong> priced peers of " + n(block.n_total) + " in " + esc(groupLabel) +
      " (coverage " + pct(block.coverage, 0) + ")" +
      (block.reliable ? "" : " — <strong>unreliable</strong>: fewer than 5 priced peers") +
      (pctText ? " &middot; this listing sits at the " + esc(pctText) : "") + "." +
      (stretched ? " The axis is stretched to " + esc(money(lo, 0)) + "–" + esc(money(hi, 0)) +
        " so this listing's price fits on it; the end labels above are still the peer group's " +
        "true " + esc(loName) + " and " + esc(hiName) + "." : "") + "</div>" +
      "</div>";
  }

  function positionSummaryHTML(block, label) {
    if (!block) return "";
    var delta = isMissing(block.delta_pct) ? null :
      (block.delta_pct >= 0 ? "+" : "") + Number(block.delta_pct).toFixed(1) + "%";
    return "<tr><td>" + esc(label) + "</td>" +
      '<td class="n">' + (isMissing(block.median) ? "—" : esc(money(block.median, 0))) + "</td>" +
      '<td class="n">' + (delta === null ? '<span class="is-missing">no price</span>' : esc(delta)) + "</td>" +
      '<td class="n">' + (isMissing(block.percentile) ? "—" : Number(block.percentile).toFixed(0)) + "</td>" +
      "<td>" + esc(block.label || "unknown") + "</td>" +
      '<td class="n">' + n(block.n) + " / " + n(block.n_total) + "</td>" +
      '<td class="n">' + pct(block.coverage, 0) + "</td>" +
      "<td>" + (block.reliable ? "yes" : '<span style="color:var(--critical)">no</span>') + "</td></tr>";
  }

  function renderOverviewTab() {
    var el = $("#panel-overview");
    if (!STATE.focusAsin) { el.innerHTML = emptyHTML("No product selected", NEED_FOCUS); return; }
    if (!STATE.detail) { el.innerHTML = loadingHTML("Loading product detail…"); return; }

    var d = STATE.detail;
    var pos = d.price_position || {};
    var seg = pos.vs_segment, brand = pos.vs_brand;
    var notes = (pos.notes || []);

    var sentBlock;
    if (d.no_sentiment_data) {
      sentBlock = '<div class="callout callout--warn"><span class="callout__icon">!</span><span>' +
        "<strong>No mined review sentiment for this product.</strong> " +
        esc(d.sentiment_note || "") + "</span></div>";
    } else {
      var s = d.sentiment;
      sentBlock = '<div class="specgrid">' +
        "<div><div class=\"spec__k\">overall polarity</div><div class=\"spec__v\">" +
        (isMissing(s.overall_polarity) ? '<span class="is-missing">not scored</span>'
          : Number(s.overall_polarity).toFixed(2) + " (" +
            esc(s.overall_polarity > 0.05 ? "positive"
              : (s.overall_polarity < -0.05 ? "negative" : "neutral")) + ")") +
        "</div></div>" +
        "<div><div class=\"spec__k\">positive share</div><div class=\"spec__v\">" + pct(s.overall_pos_share, 0) +
        "</div></div>" +
        "<div><div class=\"spec__k\">mean review rating</div><div class=\"spec__v\">" +
        (isMissing(s.mean_rating) ? "—" : Number(s.mean_rating).toFixed(2)) + "</div></div>" +
        "<div><div class=\"spec__k\">reviews scored</div><div class=\"spec__v\">" +
        n(s.n_reviews_scored) + " of " + n(d.reviews.n_reviews_retained) + " retained</div></div>" +
        "</div>";
    }

    el.innerHTML =
      '<div class="grid-2">' +
        '<section class="card"><header class="card__head"><h3>Specification</h3>' +
          '<span class="muted" style="font-size:11px">' + esc(d.parent_asin) + "</span></header>" +
          '<div class="card__body">' + specCardHTML(d) + "</div>" +
          '<div class="card__foot">Fields shown as <em>not parsed</em> were absent or unparseable in the ' +
          "source listing — they are unknown, not zero.</div></section>" +

        '<section class="card"><header class="card__head"><h3>Review sentiment</h3>' +
          (d.no_sentiment_data ? '<span class="tag">no data</span>' : '<span class="tag tag--sent">mined</span>') +
          "</header>" +
          '<div class="card__body">' + sentBlock + "</div>" +
          '<div class="card__foot">Only ' +
          (STATE.overview ? pct(STATE.overview.sentiment.coverage, 1) + " (" +
            n(STATE.overview.sentiment.n_products_with_sentiment) + " of " +
            n(STATE.overview.sentiment.n_products) + ")" : "26.5%") +
          " of the catalogue has mined sentiment; a full pass is queued.</div></section>" +
      "</div>" +

      '<section class="card"><header class="card__head"><h3>Price position vs its segment' +
        " &mdash; " + esc(d.segment) + "</h3>" +
        "<span>" + priceHTML(d.price, d.price_display) + "</span></header>" +
        '<div class="card__body">' +
        (d.price_available ? "" :
          '<div class="callout callout--warn" style="margin-bottom:12px"><span class="callout__icon">!</span>' +
          "<span><strong>This listing has no price in the source data.</strong> The peer distribution " +
          "below is still shown so the segment can be read, but no percentile exists for this product. " +
          "A missing price is unknown, not $0.</span></div>") +
        rulerHTML(seg, d.price, "the '" + d.segment + "' segment") +
        "</div>" +
        '<div class="card__body" style="border-top:1px solid var(--border)">' +
        '<div class="tablewrap"><table class="data"><caption class="visually-hidden">' +
        "Price position table view</caption><thead><tr>" +
        "<th>Peer group</th><th class=\"n\">Median</th><th class=\"n\">Δ vs median</th>" +
        "<th class=\"n\">Percentile</th><th>Label</th><th class=\"n\">n priced / total</th>" +
        "<th class=\"n\">Coverage</th><th>Reliable</th></tr></thead><tbody>" +
        positionSummaryHTML(seg, "segment · " + d.segment) +
        positionSummaryHTML(brand, "brand · " + d.brand) +
        "</tbody></table></div></div>" +
        (notes.length ? '<div class="card__foot"><ul style="margin:0;padding-left:16px">' +
          notes.map(function (t) { return "<li>" + esc(t) + "</li>"; }).join("") + "</ul></div>" : "") +
      "</section>";
  }

  /* ==================================================================== */
  /* 5. Competitors tab — ranked table, guard ablation, price/rating plot  */
  /* ==================================================================== */

  function loadCompetitors(force) {
    var asin = STATE.focusAsin;
    var k = STATE.competitors.k;
    var el = $("#panel-competitors");
    el.innerHTML = loadingHTML("Ranking competitors (matcher warm query ~30 ms)…");

    var base = "/api/products/" + encodeURIComponent(asin) + "/competitors";
    Promise.all([
      api(base + qs({ k: k, guard: "true" })),
      api(base + qs({ k: k, guard: "false" }))
    ]).then(function (res) {
      if (STATE.focusAsin !== asin) return;
      STATE.competitors.guarded = res[0];
      STATE.competitors.unguarded = res[1];
      STATE.loaded.competitors = true;
      renderCompetitorsTab(false);
    }, function (err) {
      if (STATE.focusAsin !== asin) return;
      el.innerHTML = errorHTML(err, "retryComp");
      var r = $("#retryComp");
      if (r) r.addEventListener("click", function () { loadCompetitors(true); });
    });
  }

  function competitorRowHTML(c, queryPrice, querySegment, opts) {
    var o = opts || {};
    var offSeg = c.segment !== querySegment;
    var ratio = c.similarity && !isMissing(c.similarity.price_ratio) ? c.similarity.price_ratio : null;
    var band = o.band || null;
    var outOfBand = ratio !== null && band && (ratio > band || ratio < 1 / band);
    var flags = [];
    if (offSeg) flags.push('<span class="tag" style="border-color:var(--serious)">off-segment: ' + esc(c.segment) + "</span>");
    if (outOfBand) flags.push('<span class="tag" style="border-color:var(--critical)">price ×' +
      Number(ratio).toFixed(2) + "</span>");
    if (c.is_renewed) flags.push('<span class="tag tag--renewed">renewed</span>');
    if (c.price_is_estimated) flags.push('<span class="tag">price estimated</span>');

    return '<tr' + (o.dropped ? ' class="is-dropped"' : "") + ">" +
      '<td class="n">' + n(c.rank) + "</td>" +
      '<td class="cell-title"><button type="button" class="linkish" data-asin="' + esc(c.parent_asin) +
        '"><span class="cell-title__t">' + esc(c.title) + "</span></button>" +
        (flags.length ? '<div style="margin-top:4px;display:flex;gap:4px;flex-wrap:wrap">' + flags.join("") + "</div>" : "") +
      "</td>" +
      "<td>" + esc(c.brand) + "</td>" +
      "<td>" + esc(c.segment) + "</td>" +
      '<td class="n' + (c.price_available ? "" : " is-missing") + '">' + esc(c.price_display) + "</td>" +
      '<td class="n">' + (isMissing(c.specs.ram_gb) ? "—" : n(c.specs.ram_gb)) + "</td>" +
      '<td class="n">' + (isMissing(c.average_rating) ? "—" : Number(c.average_rating).toFixed(1)) +
        '<span class="muted"> (' + n(c.rating_number) + ")</span></td>" +
      '<td class="n">' + (isMissing(c.score) ? "—" : Number(c.score).toFixed(3)) + "</td>" +
      "</tr>";
  }

  function guardEffectHTML() {
    var g = STATE.competitors.guarded, u = STATE.competitors.unguarded;
    var guardCfg = g.guard || {};
    var band = guardCfg.price_band;
    var qseg = g.query_product.segment;
    var qprice = g.query_product.price;

    var gIds = {}, uIds = {};
    g.competitors.forEach(function (c) { gIds[c.parent_asin] = c; });
    u.competitors.forEach(function (c) { uIds[c.parent_asin] = c; });

    var droppedByGuard = u.competitors.filter(function (c) { return !gIds[c.parent_asin]; });
    var promotedByGuard = g.competitors.filter(function (c) { return !uIds[c.parent_asin]; });

    function countBad(list) {
      var off = 0, oob = 0, worst = null;
      list.forEach(function (c) {
        if (c.segment !== qseg) off++;
        var r = c.similarity ? c.similarity.price_ratio : null;
        if (!isMissing(r)) {
          if (band && (r > band || r < 1 / band)) oob++;
          // "how far from the query's price tier" — symmetric, so ×0.25 counts as ×4
          var away = r >= 1 ? r : 1 / r;
          if (worst === null || away > worst) worst = away;
        }
      });
      return { off: off, oob: oob, worst: worst };
    }
    var bg = countBad(g.competitors), bu = countBad(u.competitors);
    var fmtWorst = function (w) { return w === null ? "—" : "×" + Number(w).toFixed(2); };

    // each row is pre-formatted for display; `worse` marks the guard-OFF cell in red
    var rows = [
      { label: "Off-segment matches", on: n(bg.off), off: n(bu.off),
        denom: n(g.competitors.length) + " ranked", worse: bu.off > bg.off },
      { label: "Outside the ±" + (band ? Number(band).toFixed(2) : "?") + "× price band",
        on: n(bg.oob), off: n(bu.oob),
        denom: n(g.competitors.length) + " ranked", worse: bu.oob > bg.oob },
      { label: "Widest price gap from the query product",
        on: fmtWorst(bg.worst), off: fmtWorst(bu.worst),
        denom: "lower is a tighter comparison set",
        worse: bg.worst !== null && bu.worst !== null && bu.worst > bg.worst + 0.01 },
      { label: "Rows the guard changed", on: "—",
        off: n(droppedByGuard.length) + " swapped out",
        denom: "of " + n(u.competitors.length) + " raw matches",
        worse: droppedByGuard.length > 0 }
    ];

    var verdict = (bu.off + bu.oob) > (bg.off + bg.oob) ||
      (bg.worst !== null && bu.worst !== null && bu.worst > bg.worst + 0.01) ||
      droppedByGuard.length > 0;

    return '<section class="card"><header class="card__head">' +
      "<h3>Guard ablation — what the segment + price guard removes</h3>" +
      '<label class="fld--row" style="font-size:12px;gap:6px;align-items:center">' +
      '<input type="checkbox" id="guardToggle"' + (STATE.competitors.showUnguarded ? "" : " checked") +
      "> <span>guard on</span></label></header>" +
      '<div class="card__body">' +
      '<div class="callout ' + (verdict ? "callout--good" : "callout--info") + '">' +
      '<span class="callout__icon">' + (verdict ? "✓" : "i") + "</span><span>" +
      (verdict
        ? "Turning the guard <strong>off</strong> changes <strong>" + n(droppedByGuard.length) +
          " of the " + n(u.competitors.length) + "</strong> ranked rows" +
          ((bu.off > bg.off || bu.oob > bg.oob)
            ? " and lets in <strong>" + n(bu.off) + "</strong> off-segment and <strong>" +
              n(bu.oob) + "</strong> out-of-price-band matches (guard on: " +
              n(bg.off) + " and " + n(bg.oob) + ")."
            : ", and widens the furthest price gap from " + fmtWorst(bg.worst) + " to " +
              fmtWorst(bu.worst) + " the query product's price.") +
          " Toggle it above to swap the ranked table."
        : "For this particular query the raw ranking is already clean — the guard changes no " +
          "rows in the top " + n(u.competitors.length) + ". Its value shows up on queries whose " +
          "nearest text neighbours sit in another segment or another price tier; try a mainstream " +
          "or ultrabook listing to see it bite.") +
      "</span></div>" +

      '<div class="tablewrap" style="margin-top:12px"><table class="data"><thead><tr>' +
      "<th>Contamination measure</th><th class=\"n\">Guard ON</th><th class=\"n\">Guard OFF</th>" +
      "<th class=\"n\">Denominator</th></tr></thead><tbody>" +
      rows.map(function (r) {
        return "<tr><td>" + esc(r.label) + '</td><td class="n">' + esc(r.on) + "</td>" +
          '<td class="n"' + (r.worse ? ' style="color:var(--critical);font-weight:600"' : "") +
          ">" + esc(r.off) + '</td><td class="n">' + esc(r.denom) + "</td></tr>";
      }).join("") +
      '<tr><td>Query price / segment</td><td class="n" colspan="3">' +
      (isMissing(qprice) ? '<span class="is-missing">price not listed</span>' : esc(money(qprice, 2))) +
      " &middot; " + esc(qseg) + " &middot; band ±" + (band ? Number(band).toFixed(2) : "?") +
      "×, min segment affinity " + (guardCfg.min_segment_affinity !== undefined ?
        Number(guardCfg.min_segment_affinity).toFixed(2) : "?") + "</td></tr>" +
      "</tbody></table></div>" +

      (droppedByGuard.length
        ? '<h4 style="margin:14px 0 6px;font-size:12px">Rows the guard removed (' +
          n(droppedByGuard.length) + " of " + n(u.competitors.length) + ")</h4>" +
          '<div class="tablewrap"><table class="data"><thead><tr><th class="n">#</th><th>Title</th>' +
          "<th>Brand</th><th>Segment</th><th class=\"n\">Price</th><th class=\"n\">RAM GB</th>" +
          "<th class=\"n\">Rating</th><th class=\"n\">Score</th></tr></thead><tbody>" +
          droppedByGuard.map(function (c) {
            return competitorRowHTML(c, qprice, qseg, { band: band, dropped: true });
          }).join("") + "</tbody></table></div>"
        : '<p class="muted" style="margin-top:12px;font-size:12px">The guard removed no rows from ' +
          "the top-" + n(u.competitors.length) + " for this query.</p>") +

      (promotedByGuard.length
        ? '<h4 style="margin:14px 0 6px;font-size:12px">Rows the guard promoted into the top ' +
          n(g.competitors.length) + " (" + n(promotedByGuard.length) + ")</h4>" +
          '<div class="tablewrap"><table class="data"><thead><tr><th class="n">#</th><th>Title</th>' +
          "<th>Brand</th><th>Segment</th><th class=\"n\">Price</th><th class=\"n\">RAM GB</th>" +
          "<th class=\"n\">Rating</th><th class=\"n\">Score</th></tr></thead><tbody>" +
          promotedByGuard.map(function (c) {
            return competitorRowHTML(c, qprice, qseg, { band: band });
          }).join("") + "</tbody></table></div>"
        : "") +

      "</div>" +
      '<div class="card__foot">' + esc(guardCfg.description || "") + "</div></section>";
  }

  function scatterHTML(active) {
    var priced = active.competitors.filter(function (c) { return c.price_available; });
    var unpriced = active.competitors.length - priced.length;
    var q = active.query_product;
    var qPlottable = q.price_available && !isMissing(q.average_rating);

    var note = '<div class="muted" style="font-size:11.5px">Plotting <strong class="num">' +
      n(priced.filter(function (c) { return !isMissing(c.average_rating); }).length) + "</strong> of " +
      n(active.competitors.length) + " ranked competitors: " + n(unpriced) +
      " have no listed price and " +
      n(priced.filter(function (c) { return isMissing(c.average_rating); }).length) +
      " no rating, so they cannot be placed on either axis. They remain in the table above.</div>";

    var fallback = !PLOTLY_OK();
    return '<section class="card"><header class="card__head chart__toolbar">' +
      "<h3>Price vs rating — focus product highlighted</h3>" +
      '<button type="button" class="btn btn--sm btn--ghost" id="scatterToggle"' +
      (fallback ? " disabled" : "") + ">" +
      (fallback ? "Chart unavailable — showing table" : "Show table") + "</button></header>" +
      '<div class="card__body">' +
      (fallback ? '<div class="callout callout--warn" style="margin-bottom:10px">' +
        '<span class="callout__icon">!</span><span>The Plotly CDN could not be reached, so the ' +
        "scatter is shown as its table twin. Every value below is the same data the chart would plot." +
        "</span></div>" : "") +
      '<div class="chart__legend" id="scatterLegend"' + (fallback ? ' hidden' : '') + '>' +
      '<span class="legend__item"><span class="legend__swatch" style="background:var(--series-1)"></span>ranked competitors</span>' +
      '<span class="legend__item"><span class="legend__swatch" style="background:var(--series-2)"></span>focus product' +
      (qPlottable ? "" : " (not plottable — " + (q.price_available ? "unrated" : "price not listed") + ")") +
      "</span></div>" +
      '<div class="plot" id="scatterPlot"' + (fallback ? ' hidden' : '') + '></div>' +
      '<div id="scatterTable"' + (fallback ? "" : ' hidden') + '>' + scatterTableHTML(active) + "</div>" +
      note + "</div></section>";
  }

  function scatterTableHTML(active) {
    var q = active.query_product;
    var rows = [{
      rank: "—", title: q.title, brand: q.brand, segment: q.segment,
      price: q.price, price_display: q.price_display,
      average_rating: q.average_rating, rating_number: q.rating_number, focus: true
    }].concat(active.competitors);
    return '<div class="tablewrap"><table class="data"><thead><tr>' +
      "<th class=\"n\">#</th><th>Title</th><th class=\"n\">Price (x)</th>" +
      "<th class=\"n\">Rating (y)</th><th class=\"n\">Ratings n</th></tr></thead><tbody>" +
      rows.map(function (r) {
        return "<tr" + (r.focus ? ' class="is-focus"' : "") + '><td class="n">' +
          (r.focus ? "focus" : n(r.rank)) + "</td>" +
          '<td class="cell-title"><span class="cell-title__t">' + esc(r.title) + "</span></td>" +
          '<td class="n' + (isMissing(r.price) ? " is-missing" : "") + '">' + esc(r.price_display) + "</td>" +
          '<td class="n">' + (isMissing(r.average_rating) ? '<span class="is-missing">unrated</span>'
            : Number(r.average_rating).toFixed(1)) + "</td>" +
          '<td class="n">' + n(r.rating_number) + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function drawScatter(active) {
    if (!PLOTLY_OK()) return;
    var host = $("#scatterPlot");
    if (!host) return;
    var pts = active.competitors.filter(function (c) {
      return c.price_available && !isMissing(c.average_rating);
    });
    var q = active.query_product;

    var ink = cssVar("--ink-2") || "#52514e";
    var muted = cssVar("--ink-muted") || "#898781";
    var grid = cssVar("--grid") || "#e1e0d9";
    var surface = cssVar("--surface-1") || "#fcfcfb";

    var traces = [{
      type: "scattergl", mode: "markers", name: "ranked competitors",
      x: pts.map(function (c) { return c.price; }),
      y: pts.map(function (c) { return c.average_rating; }),
      text: pts.map(function (c) {
        return truncate(c.title, 70) + "<br>" + c.brand + " · " + c.segment +
          "<br>" + c.price_display + " · " + Number(c.average_rating).toFixed(1) +
          "★ (n=" + c.rating_number + ")<br>rank " + c.rank + " · score " +
          (isMissing(c.score) ? "n/a" : Number(c.score).toFixed(3));
      }),
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        size: 11, color: cssVar("--series-1") || "#2a78d6", opacity: 0.9,
        line: { width: 2, color: surface }
      }
    }];

    if (q.price_available && !isMissing(q.average_rating)) {
      traces.push({
        type: "scattergl", mode: "markers+text", name: "focus product",
        x: [q.price], y: [q.average_rating],
        text: ["focus"], textposition: "top center",
        textfont: { color: ink, size: 11 },
        hovertext: [truncate(q.title, 70) + "<br>" + q.brand + " · " + q.segment + "<br>" +
          q.price_display + " · " + Number(q.average_rating).toFixed(1) + "★"],
        hovertemplate: "%{hovertext}<extra></extra>",
        marker: {
          size: 18, symbol: "diamond", color: cssVar("--series-2") || "#eb6834",
          line: { width: 2, color: surface }
        }
      });
    }

    var layout = {
      margin: { l: 54, r: 16, t: 10, b: 46 },
      paper_bgcolor: surface, plot_bgcolor: surface,
      showlegend: false,
      hovermode: "closest",
      font: { family: "system-ui, -apple-system, 'Segoe UI', sans-serif", size: 11, color: ink },
      xaxis: {
        title: { text: "listed price (USD, log scale)", font: { size: 11, color: muted } },
        type: "log", gridcolor: grid, zeroline: false, linecolor: grid,
        tickfont: { color: muted }
      },
      yaxis: {
        title: { text: "average rating", font: { size: 11, color: muted } },
        gridcolor: grid, zeroline: false, linecolor: grid, range: [0.8, 5.2],
        tickfont: { color: muted }
      }
    };
    try {
      window.Plotly.newPlot(host, traces, layout,
        { displayModeBar: false, responsive: true, doubleClick: "reset" });
      host.on("plotly_click", function (ev) {
        var p = ev.points && ev.points[0];
        if (!p || p.curveNumber !== 0) return;
        var c = pts[p.pointNumber];
        if (c) setFocus(c.parent_asin);
      });
    } catch (e) {
      host.hidden = true;
      var t = $("#scatterTable");
      if (t) t.hidden = false;
      var b = $("#scatterToggle");
      if (b) { b.textContent = "Chart unavailable"; b.disabled = true; }
      var lg = $("#scatterLegend");
      if (lg) lg.hidden = true;
    }
  }

  function renderCompetitorsTab(force) {
    var el = $("#panel-competitors");
    if (!STATE.focusAsin) { el.innerHTML = emptyHTML("No product selected", NEED_FOCUS); return; }
    if (!STATE.loaded.competitors || force) {
      if (!STATE.competitors.guarded || force) { loadCompetitors(force); return; }
    }
    var g = STATE.competitors.guarded, u = STATE.competitors.unguarded;
    if (!g) { el.innerHTML = loadingHTML("Ranking competitors…"); return; }

    var showUn = !!STATE.competitors.showUnguarded;
    var active = showUn ? u : g;
    var q = g.query_product;
    var band = (g.guard || {}).price_band;

    el.innerHTML =
      '<section class="card"><header class="card__head">' +
        "<h3>Top " + n(active.competitors.length) + " competitors — guard " +
        (showUn ? '<span style="color:var(--critical)">OFF (ablation)</span>' : "ON") + "</h3>" +
        '<span class="muted" style="font-size:11.5px">' + n(active.took_ms, 0) + " ms &middot; " +
        n(active.coverage.n_priced) + " of " + n(active.competitors.length) +
        " priced (" + pct(active.coverage.price_coverage, 0) + ")</span></header>" +
        '<div class="card__body card__body--tight"><div class="tablewrap"><table class="data">' +
        "<thead><tr><th class=\"n\">#</th><th>Title</th><th>Brand</th><th>Segment</th>" +
        "<th class=\"n\">Price</th><th class=\"n\">RAM GB</th><th class=\"n\">Rating</th>" +
        "<th class=\"n\">Score</th></tr></thead><tbody>" +
        active.competitors.map(function (c) {
          return competitorRowHTML(c, q.price, q.segment, { band: band });
        }).join("") +
        "</tbody></table></div></div>" +
        '<div class="card__foot">Score is the hybrid text + spec similarity from matching.py. ' +
        "Rows with no listed price still rank: the matcher substitutes a hierarchical price " +
        "estimate for the guard only, and the table shows &ldquo;price not listed&rdquo; rather " +
        "than a fabricated figure.</div></section>" +

      guardEffectHTML() +
      scatterHTML(active);

    var gt = $("#guardToggle");
    if (gt) {
      gt.addEventListener("change", function () {
        STATE.competitors.showUnguarded = !gt.checked;
        renderCompetitorsTab(false);
      });
    }
    var st = $("#scatterToggle");
    if (st && PLOTLY_OK()) {
      st.addEventListener("click", function () {
        var plot = $("#scatterPlot"), tbl = $("#scatterTable"), lg = $("#scatterLegend");
        var showTable = plot.hidden === false;
        plot.hidden = showTable;
        if (lg) lg.hidden = showTable;
        tbl.hidden = !showTable;
        st.textContent = showTable ? "Show chart" : "Show table";
      });
    }
    $$("#panel-competitors .linkish").forEach(function (b) {
      b.addEventListener("click", function () { setFocus(b.getAttribute("data-asin")); });
    });

    drawScatter(active);
  }

  /* ==================================================================== */
  /* 6. Pricing tab — segment & brand positioning, every figure with n     */
  /* ==================================================================== */

  /**
   * Horizontal bars. Values may be signed (the discrete-GPU premium is negative
   * wherever the discrete median sits *below* the integrated one). A negative
   * value must never be fed to `width:` — CSS rejects a negative length, the
   * declaration is dropped and the bar silently collapses to its 2px min-width,
   * which reads as "no effect" and contradicts the printed label. So when any
   * row is negative the whole chart switches to a zero-midpoint diverging
   * layout: magnitude is scaled against the largest |value| and direction is
   * carried by both the side of the midline and the diverging colour pair.
   */
  function hbarsHTML(rows, opts) {
    var o = opts || {};
    var max = 0, signed = false;
    rows.forEach(function (r) {
      if (isMissing(r.value)) return;
      if (r.value < 0) signed = true;
      var m = Math.abs(r.value);
      if (m > max) max = m;
    });
    if (max <= 0) max = 1;

    return '<div class="hbars" style="--label-w:' + (o.labelWidth || 116) + 'px">' +
      rows.map(function (r) {
        var v = isMissing(r.value) ? null : Number(r.value);
        var style, colour;
        if (v === null) {
          // no fill at all: .hbar__fill has min-width:2px, so a "zero-width" bar
          // still paints a stub — at left:0 that reads as a large negative once
          // the chart is diverging. An unmeasured row must draw nothing.
          style = null;
          colour = "";
        } else if (!signed) {
          style = "left:0;width:" + ((v / max) * 100).toFixed(2) + "%";
          colour = r.highlight ? ";background:var(--series-2)" : "";
        } else {
          // half-width each side of a 50% zero line
          var half = ((Math.abs(v) / max) * 50).toFixed(2);
          style = v >= 0 ? "left:50%;width:" + half + "%"
                         : "left:auto;right:50%;width:" + half + "%";
          colour = r.highlight ? ";background:var(--series-2)"
            : ";background:var(--div-" + (v >= 0 ? "pos" : "neg") + ")";
        }
        return '<div class="hbar' + (r.unreliable ? " is-unreliable" : "") + '">' +
          '<div class="hbar__label" title="' + esc(r.label) + '">' + esc(r.label) + "</div>" +
          '<div class="hbar__track" data-tip="' + esc(r.tip || "") + '">' +
          (style === null ? "" : '<div class="hbar__fill" style="' + style + colour + '"></div>') +
          (signed ? '<div class="dbar__mid"></div>' : "") + "</div>" +
          '<div class="hbar__value">' + (r.valueLabel === undefined ? n(r.value) : r.valueLabel) + "</div>" +
          (r.sub ? '<div class="hbar__sub">' + esc(r.sub) + "</div>" : "") +
          "</div>";
      }).join("") + "</div>";
  }

  function segmentChartHTML() {
    var seg = STATE.segments;
    if (!seg) return errorHTML({ message: "Segment table not loaded." });
    var focusSeg = STATE.detail ? STATE.detail.segment : null;
    var rows = seg.rows.slice().sort(function (a, b) {
      return (b.median || 0) - (a.median || 0);
    }).map(function (r) {
      return {
        label: r.segment, value: r.median,
        valueLabel: isMissing(r.median) ? '<span class="is-missing">no priced rows</span>' : money(r.median, 0),
        unreliable: r.unreliable,
        highlight: r.segment === focusSeg,
        sub: "n=" + n(r.n_priced) + " priced of " + n(r.n_products) + " listings · coverage " +
          pct(r.coverage, 0) + (r.unreliable ? " · UNRELIABLE (<5 priced)" : ""),
        tip: r.segment + "\nmedian " + (money(r.median, 0) || "n/a") +
          "\np25 " + (money(r.p25, 0) || "n/a") + " · p75 " + (money(r.p75, 0) || "n/a") +
          "\nn=" + r.n_priced + " priced of " + r.n_products
      };
    });

    return '<section class="card"><header class="card__head">' +
      "<h3>Median price by segment</h3>" +
      '<span class="muted" style="font-size:11.5px">overall price coverage ' +
      pct(seg.overall_price_coverage, 1) + " across " + n(seg.n_products) + " listings</span></header>" +
      '<div class="card__body">' +
      (focusSeg ? '<div class="chart__legend" style="margin-bottom:10px">' +
        '<span class="legend__item"><span class="legend__swatch" style="background:var(--series-1)"></span>segment median</span>' +
        '<span class="legend__item"><span class="legend__swatch" style="background:var(--series-2)"></span>focus product\'s segment (' +
        esc(focusSeg) + ")</span></div>" : "") +
      hbarsHTML(rows, { labelWidth: 104 }) +
      '<details style="margin-top:14px"><summary style="cursor:pointer;font-size:12px;color:var(--ink-2)">Table view — every figure with its n and coverage</summary>' +
      '<div class="tablewrap" style="margin-top:8px"><table class="data"><thead><tr><th>Segment</th>' +
      '<th class="n">Listings</th><th class="n">Priced</th><th class="n">Coverage</th>' +
      '<th class="n">p25</th><th class="n">Median</th><th class="n">p75</th>' +
      '<th class="n">Mean rating</th><th class="n">Sentiment cov.</th><th>Reliable</th>' +
      "</tr></thead><tbody>" +
      seg.rows.map(function (r) {
        return "<tr" + (r.segment === focusSeg ? ' class="is-focus"' : "") + "><td>" + esc(r.segment) + "</td>" +
          '<td class="n">' + n(r.n_products) + '</td><td class="n">' + n(r.n_priced) + "</td>" +
          '<td class="n">' + pct(r.coverage, 1) + "</td>" +
          '<td class="n">' + (money(r.p25, 0) || "—") + '</td><td class="n">' + (money(r.median, 0) || "—") + "</td>" +
          '<td class="n">' + (money(r.p75, 0) || "—") + "</td>" +
          '<td class="n">' + (isMissing(r.mean_rating) ? "—" : Number(r.mean_rating).toFixed(2)) + "</td>" +
          '<td class="n">' + pct(r.sentiment_coverage, 1) + "</td>" +
          "<td>" + (r.reliable ? "yes" : '<span style="color:var(--critical)">no</span>') + "</td></tr>";
      }).join("") + "</tbody></table></div></details>" +
      "</div>" +
      '<div class="card__foot">' + esc((seg.notes || []).join(" ")) + "</div></section>";
  }

  function brandChartHTML() {
    var br = STATE.brands;
    if (!br) return errorHTML({ message: "Brand table not loaded." });
    var focusBrand = STATE.detail ? STATE.detail.brand : null;
    var top = br.rows.slice(0, 14);
    if (focusBrand && !top.some(function (r) { return r.brand === focusBrand; })) {
      var hit = br.rows.filter(function (r) { return r.brand === focusBrand; })[0];
      if (hit) top = top.slice(0, 13).concat([hit]);
    }
    var rows = top.slice().sort(function (a, b) { return (b.median || 0) - (a.median || 0); })
      .map(function (r) {
        return {
          label: r.brand, value: r.median,
          valueLabel: isMissing(r.median) ? '<span class="is-missing">no priced rows</span>' : money(r.median, 0),
          unreliable: r.unreliable,
          highlight: r.brand === focusBrand,
          sub: "n=" + n(r.n_priced) + " priced of " + n(r.n_products) + " listings · coverage " +
            pct(r.coverage, 0) + (r.unreliable ? " · UNRELIABLE (<" + n(br.min_reliable_n) + " priced)" : ""),
          tip: r.brand + "\nmedian " + (money(r.median, 0) || "n/a") +
            "\nn=" + r.n_priced + " priced of " + r.n_products +
            "\nmean rating " + (isMissing(r.mean_rating) ? "n/a" : Number(r.mean_rating).toFixed(2))
        };
      });

    return '<section class="card"><header class="card__head">' +
      "<h3>Median price by brand — " + n(rows.length) + " largest by catalogue size</h3>" +
      '<span class="muted" style="font-size:11.5px">' + n(br.n_unreliable) + " of " + n(br.n_brands) +
      " brands have fewer than " + n(br.min_reliable_n) + " priced listings</span></header>" +
      '<div class="card__body">' +
      (focusBrand ? '<div class="chart__legend" style="margin-bottom:10px">' +
        '<span class="legend__item"><span class="legend__swatch" style="background:var(--series-1)"></span>brand median</span>' +
        '<span class="legend__item"><span class="legend__swatch" style="background:var(--series-2)"></span>focus product\'s brand (' +
        esc(focusBrand) + ")</span></div>" : "") +
      hbarsHTML(rows, { labelWidth: 104 }) +
      '<details style="margin-top:14px"><summary style="cursor:pointer;font-size:12px;color:var(--ink-2)">Table view — every figure with its n and coverage</summary>' +
      '<div class="tablewrap" style="margin-top:8px"><table class="data"><thead><tr><th>Brand</th>' +
      '<th class="n">Listings</th><th class="n">Priced</th><th class="n">Coverage</th>' +
      '<th class="n">p25</th><th class="n">Median</th><th class="n">p75</th>' +
      '<th class="n">Mean rating</th><th>Reliable</th></tr></thead><tbody>' +
      top.map(function (r) {
        return "<tr" + (r.brand === focusBrand ? ' class="is-focus"' : "") + "><td>" + esc(r.brand) + "</td>" +
          '<td class="n">' + n(r.n_products) + '</td><td class="n">' + n(r.n_priced) + "</td>" +
          '<td class="n">' + pct(r.coverage, 1) + "</td>" +
          '<td class="n">' + (money(r.p25, 0) || "—") + '</td><td class="n">' + (money(r.median, 0) || "—") + "</td>" +
          '<td class="n">' + (money(r.p75, 0) || "—") + "</td>" +
          '<td class="n">' + (isMissing(r.mean_rating) ? "—" : Number(r.mean_rating).toFixed(2)) + "</td>" +
          "<td>" + (r.reliable ? "yes" : '<span style="color:var(--critical)">no</span>') + "</td></tr>";
      }).join("") + "</tbody></table></div></details>" +
      "</div>" +
      '<div class="card__foot">Brands with zero price coverage are kept in the table rather than ' +
      "silently deleted; their median reads &ldquo;no priced rows&rdquo;.</div></section>";
  }

  function gpuPremiumHTML() {
    var ov = STATE.overview;
    if (!ov) return "";
    var gp = ov.discrete_gpu_premium;
    if (!gp || gp.unavailable || !gp.overall) {
      return '<section class="card"><header class="card__head"><h3>Discrete-GPU price premium</h3></header>' +
        '<div class="card__body"><div class="callout callout--warn"><span class="callout__icon">!</span>' +
        "<span>Not available: " + esc((gp && gp.unavailable) || "no data") + "</span></div></div></section>";
    }
    var o = gp.overall;
    var rows = (gp.by_segment || []).slice().sort(function (a, b) {
      return (b.premium_pct || 0) - (a.premium_pct || 0);
    });
    var focusSeg = STATE.detail ? STATE.detail.segment : null;
    // a negative premium is a real result (discrete chromebooks are cheaper), so the
    // chart becomes diverging and needs a legend naming the zero line
    var anyNegative = rows.some(function (r) {
      return !isMissing(r.premium_pct) && r.premium_pct < 0;
    });

    return '<section class="card"><header class="card__head"><h3>Discrete-GPU price premium by segment</h3>' +
      '<span class="muted" style="font-size:11.5px">catalogue-wide: ' + esc(money(o.premium_usd, 0)) +
      " (" + n(o.premium_pct, 0) + "%) on n=" + n(o.n_discrete) + " discrete vs " +
      n(o.n_integrated) + " integrated priced listings</span></header>" +
      '<div class="card__body">' +
      (anyNegative ? '<div class="chart__legend" style="margin-bottom:10px">' +
        '<span class="legend__item"><span class="legend__swatch" style="background:var(--div-pos)"></span>discrete costs more</span>' +
        '<span class="legend__item"><span class="legend__swatch" style="background:var(--div-neg)"></span>discrete costs <em>less</em></span>' +
        '<span class="legend__item"><span class="legend__swatch" style="background:var(--axis)"></span>0% (no premium)</span>' +
        "</div>" : "") +
      hbarsHTML(rows.map(function (r) {
        return {
          label: r.segment, value: r.premium_pct,
          valueLabel: isMissing(r.premium_pct) ? "—" : (r.premium_pct >= 0 ? "+" : "") + n(r.premium_pct, 0) + "%",
          unreliable: !r.reliable,
          highlight: r.segment === focusSeg,
          sub: "median " + (money(r.median_discrete, 0) || "—") + " discrete (n=" + n(r.n_discrete) +
            ") vs " + (money(r.median_integrated, 0) || "—") + " integrated (n=" + n(r.n_integrated) +
            ") · coverage " + pct(r.coverage, 0) + (r.reliable ? "" : " · UNRELIABLE"),
          tip: r.segment + "\ndiscrete median " + (money(r.median_discrete, 0) || "n/a") +
            "\nintegrated median " + (money(r.median_integrated, 0) || "n/a") +
            "\nn=" + r.n_discrete + " vs " + r.n_integrated + " priced"
        };
      }), { labelWidth: 104 }) +
      "</div>" +
      '<div class="card__foot">Premium is the difference of medians over priced listings only ' +
      "(" + n(o.n_priced) + " of " + n(o.n_products) + " listings priced, " + pct(o.coverage, 1) +
      "). Bars at reduced opacity have too few priced rows to be reliable.</div></section>";
  }

  function renderPricingTab(force) {
    var el = $("#panel-pricing");
    var need = [];
    if (!STATE.segments) need.push(api("/api/segments").then(function (d) { STATE.segments = d; }));
    if (!STATE.brands) need.push(api("/api/brands" + qs({ limit: 300, sort: "n_products", desc: true }))
      .then(function (d) { STATE.brands = d; }));
    if (!STATE.overview) need.push(loadOverview().catch(function () { }));

    if (need.length) {
      el.innerHTML = loadingHTML("Loading segment and brand price tables…");
      Promise.all(need).then(function () { renderPricingTab(false); }, function (err) {
        el.innerHTML = errorHTML(err, "retryPricing");
        var r = $("#retryPricing");
        if (r) r.addEventListener("click", function () { renderPricingTab(true); });
      });
      return;
    }

    var caveats = STATE.overview ? (STATE.overview.caveats || []) : [];
    el.innerHTML =
      (caveats.length ? '<div class="callout callout--info" style="margin-bottom:12px">' +
        '<span class="callout__icon">i</span><span><strong>Read every figure on this tab with its ' +
        "denominator.</strong><ul style=\"margin:5px 0 0;padding-left:16px\">" +
        caveats.map(function (c) { return "<li>" + esc(c) + "</li>"; }).join("") +
        "</ul></span></div>" : "") +
      segmentChartHTML() + brandChartHTML() + gpuPremiumHTML();
  }

  /* ==================================================================== */
  /* 7. Reviews tab — aspect polarity + verbatim snippets                  */
  /* ==================================================================== */

  function aspectLabel(a) { return String(a).replace(/_/g, " "); }

  function aspectChartHTML(rv) {
    var rows = (rv.aspects || []).filter(function (a) { return !a.error && !isMissing(a.pos_share); });
    if (!rows.length) {
      return '<div class="callout callout--warn"><span class="callout__icon">!</span><span>' +
        "Sentiment was mined for this product but no aspect passed the mention threshold, so " +
        "there is nothing to plot. The verbatim snippets below are still available.</span></div>";
    }
    rows = rows.slice().sort(function (a, b) { return b.mentions - a.mentions; });

    var bars = rows.map(function (a) {
      var v = a.pos_share;
      var isPos = v >= 0.5;
      var half = Math.abs(v - 0.5) * 100;          // 0..50 -> % of the full track
      var style = isPos
        ? "left:50%;width:" + half.toFixed(2) + "%"
        : "right:50%;width:" + half.toFixed(2) + "%";
      return '<div class="hbar">' +
        '<div class="hbar__label" title="' + esc(aspectLabel(a.aspect)) + '">' +
        esc(aspectLabel(a.aspect)) + "</div>" +
        '<div class="dbar__track" data-tip="' + esc(
          aspectLabel(a.aspect) + "\npositive share " + pct(a.pos_share, 0) +
          "\nmean polarity " + (isMissing(a.polarity) ? "n/a" : Number(a.polarity).toFixed(2)) +
          "\n" + a.mentions + " mention" + (a.mentions === 1 ? "" : "s") +
          " in " + rv.n_reviews_scored + " scored reviews") + '">' +
        '<div class="dbar__fill ' + (isPos ? "dbar__fill--pos" : "dbar__fill--neg") +
        '" style="' + style + '"></div><div class="dbar__mid"></div></div>' +
        '<div class="hbar__value">' + pct(a.pos_share, 0) + "</div>" +
        '<div class="hbar__sub">' + esc(a.mentions + " mention" + (a.mentions === 1 ? "" : "s") +
          " · polarity " + (isMissing(a.polarity) ? "n/a" : Number(a.polarity).toFixed(2)) +
          " · " + (a.label || "")) + "</div>" +
        "</div>";
    }).join("");

    return '<div class="chart">' +
      '<div class="chart__legend">' +
      '<span class="legend__item"><span class="legend__swatch" style="background:var(--div-pos)"></span>above 50% positive</span>' +
      '<span class="legend__item"><span class="legend__swatch" style="background:var(--div-neg)"></span>below 50% positive</span>' +
      '<span class="legend__item"><span class="legend__swatch" style="background:var(--axis)"></span>50% midpoint</span>' +
      "</div>" +
      '<div class="hbars" style="--label-w:132px">' + bars + "</div>" +
      '<details style="margin-top:6px"><summary style="cursor:pointer;font-size:12px;color:var(--ink-2)">Table view</summary>' +
      '<div class="tablewrap" style="margin-top:8px"><table class="data"><thead><tr><th>Aspect</th>' +
      '<th class="n">Mentions</th><th class="n">Positive share</th><th class="n">Mean polarity</th>' +
      '<th>Label</th><th class="n">Snippets +/−</th></tr></thead><tbody>' +
      rows.map(function (a) {
        return "<tr><td>" + esc(aspectLabel(a.aspect)) + '</td><td class="n">' + n(a.mentions) + "</td>" +
          '<td class="n">' + pct(a.pos_share, 0) + '</td><td class="n">' +
          (isMissing(a.polarity) ? "—" : Number(a.polarity).toFixed(3)) + "</td>" +
          "<td>" + esc(a.label || "—") + '</td><td class="n">' + n(a.n_praises) + " / " + n(a.n_complaints) +
          "</td></tr>";
      }).join("") + "</tbody></table></div></details></div>";
  }

  function snippetHTML(s, kind) {
    return '<div class="snip snip--' + kind + '">' +
      '<div class="snip__text">&ldquo;' + esc(s.snippet) + "&rdquo;</div>" +
      '<div class="snip__meta">' +
      "<span>" + esc(aspectLabel(s.aspect)) + "</span>" +
      (isMissing(s.polarity) ? "" : "<span>polarity " + Number(s.polarity).toFixed(2) + "</span>") +
      (isMissing(s.rating) ? "" : "<span>" + Number(s.rating).toFixed(0) + "★ review</span>") +
      (isMissing(s.review_year) ? "" : "<span>" + n(s.review_year) + "</span>") +
      (s.verified_purchase ? "<span>verified purchase</span>" : "<span>unverified purchase</span>") +
      (isMissing(s.helpful_vote) ? "" : "<span>" + n(s.helpful_vote) + " helpful</span>") +
      "</div></div>";
  }

  function renderReviewsTab(force) {
    var el = $("#panel-reviews");
    if (!STATE.focusAsin) { el.innerHTML = emptyHTML("No product selected", NEED_FOCUS); return; }

    if (!STATE.reviews || force || STATE.reviews.parent_asin !== STATE.focusAsin) {
      var asin = STATE.focusAsin;
      el.innerHTML = loadingHTML("Loading mined review evidence…");
      api("/api/products/" + encodeURIComponent(asin) + "/reviews" + qs({ k: 3 }))
        .then(function (rv) {
          if (STATE.focusAsin !== asin) return;
          STATE.reviews = rv;
          renderReviewsTab(false);
        }, function (err) {
          if (STATE.focusAsin !== asin) return;
          el.innerHTML = errorHTML(err, "retryReviews");
          var r = $("#retryReviews");
          if (r) r.addEventListener("click", function () { renderReviewsTab(true); });
        });
      return;
    }

    var rv = STATE.reviews;

    if (rv.no_sentiment_data) {
      el.innerHTML = '<section class="card"><header class="card__head">' +
        "<h3>Review sentiment</h3><span class=\"tag\">not mined yet</span></header>" +
        '<div class="card__body">' +
        '<div class="callout callout--warn"><span class="callout__icon">!</span><span>' +
        "<strong>This product has no mined review sentiment, so there is nothing to chart.</strong> " +
        esc(rv.sentiment_note || "") + "</span></div>" +
        '<div class="specgrid" style="margin-top:14px">' +
        "<div><div class=\"spec__k\">reviews retained</div><div class=\"spec__v\">" +
        n(rv.n_reviews_retained) + "</div></div>" +
        "<div><div class=\"spec__k\">Amazon ratings</div><div class=\"spec__v\">" +
        n(rv.rating_number) + "</div></div>" +
        "<div><div class=\"spec__k\">reviews scored</div><div class=\"spec__v is-missing\">0 — pass queued</div></div>" +
        "</div></div>" +
        '<div class="card__foot">Rendering a zeroed chart here would imply neutral sentiment. ' +
        "There is no measurement, which is a different thing.</div></section>";
      return;
    }

    var ov = rv.overall || {};
    var praises = rv.praises || [], complaints = rv.complaints || [];

    el.innerHTML =
      '<section class="card"><header class="card__head"><h3>Aspect sentiment — positive share</h3>' +
        '<span class="muted" style="font-size:11.5px">' + n(rv.n_reviews_scored) +
        " reviews scored of " + n(rv.n_reviews_retained) + " retained</span></header>" +
        '<div class="card__body">' +
        '<div class="specgrid" style="margin-bottom:14px">' +
        "<div><div class=\"spec__k\">overall polarity</div><div class=\"spec__v\">" +
        (isMissing(ov.polarity) ? "—" : Number(ov.polarity).toFixed(2)) + " (" + esc(ov.label || "—") +
        ")</div></div>" +
        "<div><div class=\"spec__k\">overall positive share</div><div class=\"spec__v\">" +
        pct(ov.pos_share, 0) + "</div></div>" +
        "<div><div class=\"spec__k\">mean review rating</div><div class=\"spec__v\">" +
        (isMissing(ov.mean_rating) ? "—" : Number(ov.mean_rating).toFixed(2)) + "</div></div>" +
        "<div><div class=\"spec__k\">aspects with mentions</div><div class=\"spec__v\">" +
        n((rv.aspects || []).length) + " of 8</div></div>" +
        "</div>" +
        aspectChartHTML(rv) +
        "</div>" +
        '<div class="card__foot">Positive share is measured over the clauses that mention each ' +
        "aspect, so a small mention count is a weak signal — the count is printed beside every " +
        "bar for that reason.</div></section>" +

      '<div class="grid-2">' +
        '<section class="card"><header class="card__head"><h3>What reviewers praise</h3>' +
        '<span class="tag">' + n(praises.length) + " snippets</span></header>" +
        '<div class="card__body">' +
        (praises.length ? '<div class="snips">' + praises.map(function (s) { return snippetHTML(s, "pos"); }).join("") + "</div>"
          : '<div class="callout"><span class="callout__icon">i</span><span>No positive snippet ' +
            "cleared the opinion-strength threshold for this product.</span></div>") +
        "</div></section>" +

        '<section class="card"><header class="card__head"><h3>What reviewers complain about</h3>' +
        '<span class="tag">' + n(complaints.length) + " snippets</span></header>" +
        '<div class="card__body">' +
        (complaints.length ? '<div class="snips">' + complaints.map(function (s) { return snippetHTML(s, "neg"); }).join("") + "</div>"
          : '<div class="callout"><span class="callout__icon">i</span><span>No negative snippet ' +
            "cleared the opinion-strength threshold for this product.</span></div>") +
        "</div></section>" +
      "</div>" +

      '<section class="card"><header class="card__head"><h3>Per-aspect verbatim evidence</h3></header>' +
        '<div class="card__body">' +
        (rv.aspects || []).map(function (a) {
          if (a.error) {
            return '<div class="callout callout--warn" style="margin-bottom:9px"><span class="callout__icon">!</span>' +
              "<span>" + esc(aspectLabel(a.aspect)) + ": " + esc(a.error) + "</span></div>";
          }
          var body = ((a.praises || []).map(function (s) { return snippetHTML(s, "pos"); })
            .concat((a.complaints || []).map(function (s) { return snippetHTML(s, "neg"); }))).join("");
          return "<details style=\"margin-bottom:8px\"><summary style=\"cursor:pointer;font-size:12.5px\">" +
            "<strong>" + esc(aspectLabel(a.aspect)) + "</strong> — " + n(a.mentions) +
            " mentions, " + pct(a.pos_share, 0) + " positive, " +
            n(a.n_praises) + " praise / " + n(a.n_complaints) + " complaint snippets</summary>" +
            '<div class="snips" style="margin-top:8px">' +
            (body || '<p class="muted" style="font-size:12px">No snippet passed the threshold for this aspect.</p>') +
            "</div></details>";
        }).join("") +
        "</div>" +
        '<div class="card__foot">' + esc(rv.note || "") + "</div></section>";
  }

  /* ==================================================================== */
  /* 8. Chat — grounded RAG with inspectable citations and an audit badge  */
  /* ==================================================================== */

  function llmPillRender(st, err) {
    var pill = $("#llmPill");
    var notice = $("#chatNotice");
    var input = $("#chatInput");
    var send = $("#chatSend");
    var hint = $("#chatHint");

    if (err) {
      pill.className = "pill pill--bad";
      pill.textContent = "status unknown";
      notice.hidden = false;
      notice.innerHTML = "Could not read the model status: " + esc(err.message) +
        " — sending a question will still be attempted.";
      return;
    }
    STATE.chat.llm = st;
    var map = {
      unloaded: ["pill--muted", "model idle"],
      loading: ["pill--busy", "model loading…"],
      ready: ["pill--ok", "model ready"],
      unavailable: ["pill--bad", "model unavailable"]
    };
    var m = map[st.state] || ["pill--muted", st.state];
    pill.className = "pill " + m[0];
    pill.textContent = m[1];
    pill.title = st.model + (st.load_seconds ? " · loaded in " + st.load_seconds + " s" : "") +
      " · answers served " + st.answers_served;

    var blocked = STATE.chat.busy || st.state === "loading" || st.state === "unavailable";
    input.disabled = blocked;
    send.disabled = blocked;

    if (st.state === "loading") {
      notice.hidden = false;
      notice.innerHTML = '<span class="spinner"></span> <strong>Input disabled:</strong> the ' +
        esc(st.model) + " weights are being loaded onto the GPU (~30–60 s). " +
        "It happens once per server process.";
      hint.textContent = "waiting for the model…";
    } else if (st.state === "unavailable") {
      notice.hidden = false;
      notice.innerHTML = "<strong>Input disabled:</strong> the model could not be loaded — " +
        esc(st.error || "unknown error") + ". " +
        (st.gpu && st.gpu.memory_used_mib !== null
          ? "GPU currently holds " + n(st.gpu.memory_used_mib) + " MiB of " +
            n(st.gpu.memory_total_mib) + " MiB." : "");
      hint.textContent = "chat disabled";
    } else if (STATE.chat.busy) {
      notice.hidden = false;
      notice.innerHTML = '<span class="spinner"></span> Generating an answer — one question at a ' +
        "time on a single GPU.";
      hint.textContent = "thinking…";
    } else if (st.state === "unloaded") {
      notice.hidden = false;
      notice.innerHTML = "The 7B model is not resident yet: your first question also loads it " +
        "(~30–60 s), after that answers take about 2–17 s." +
        (st.gpu && st.gpu.memory_used_mib !== null && !st.gpu.can_load
          ? " <strong>The GPU is currently busy (" + n(st.gpu.memory_used_mib) +
            " MiB in use); loading will wait rather than risk an out-of-memory crash.</strong>" : "");
      hint.textContent = "Enter to send · Shift+Enter for a new line";
    } else {
      notice.hidden = true;
      notice.innerHTML = "";
      hint.textContent = "Enter to send · Shift+Enter for a new line";
    }
  }

  function pollChatStatus() {
    return api("/api/chat/status").then(function (st) {
      llmPillRender(st, null);
      return st;
    }, function (err) {
      llmPillRender(null, err);
      throw err;
    });
  }

  function startStatusPolling() {
    if (STATE.chat.pollTimer) return;
    STATE.chat.pollTimer = setInterval(function () {
      pollChatStatus().then(function (st) {
        if (st.state !== "loading" && !STATE.chat.busy) {
          clearInterval(STATE.chat.pollTimer);
          STATE.chat.pollTimer = null;
        }
      }, function () { });
    }, 3000);
  }

  function chatScroll() {
    var log = $("#chatLog");
    log.scrollTop = log.scrollHeight;
  }

  function appendUserMessage(text) {
    var log = $("#chatLog");
    var intro = $(".chat__intro", log);
    if (intro) intro.remove();
    var div = document.createElement("div");
    div.className = "msg msg--user";
    div.innerHTML = '<div class="msg__bubble">' + esc(text) + "</div>";
    log.appendChild(div);
    chatScroll();
  }

  function appendThinking(id) {
    var log = $("#chatLog");
    var div = document.createElement("div");
    div.className = "msg msg--bot";
    div.id = id;
    div.innerHTML = '<div class="msg__bubble"><div class="chat__thinking">' +
      '<span class="spinner"></span><span id="' + id + '-t">Retrieving evidence…</span></div></div>';
    log.appendChild(div);
    chatScroll();

    var t0 = Date.now();
    var label = $("#" + id + "-t");
    var timer = setInterval(function () {
      if (!document.getElementById(id + "-t")) { clearInterval(timer); return; }
      var s = Math.round((Date.now() - t0) / 1000);
      var phase = (STATE.chat.llm && STATE.chat.llm.state === "ready")
        ? "Retrieving evidence and generating a grounded answer"
        : "Loading the model, then generating";
      label.textContent = phase + "… " + s + " s";
    }, 500);
    return function () { clearInterval(timer); };
  }

  /** Turn [P1]/[R2]/[S3] into clickable buttons scoped to one message. */
  function linkifyCitations(answer, msgId, evidenceByMarker) {
    return esc(answer).replace(/\[([PRS])(\d{1,3})\]/g, function (whole, kind, num) {
      var key = kind + num;
      var known = Object.prototype.hasOwnProperty.call(evidenceByMarker, key);
      return '<button type="button" class="cite' + (known ? "" : " cite--bad") +
        '" data-msg="' + msgId + '" data-marker="' + key + '"' +
        (known ? "" : ' title="This citation marker matches no evidence block — flagged by the audit."') +
        ">[" + key + "]</button>";
    });
  }

  function evidenceHTML(ev, msgId) {
    var m = ev.meta || {};
    var marker = String(ev.marker || "").replace(/[[\]]/g, "");
    var id = "ev-" + msgId + "-" + marker;

    if (ev.kind === "product") {
      return '<div class="ev ev--product" id="' + id + '" tabindex="-1">' +
        '<div class="ev__head"><span class="ev__marker">' + esc(ev.marker) + "</span>" +
        '<span class="ev__title">' + esc(truncate(m.title || "(untitled)", 110)) + "</span></div>" +
        '<div class="ev__meta">' +
        "<span>" + esc(m.brand || "—") + "</span><span>" + esc(m.segment || "—") + "</span>" +
        "<span>" + (isMissing(m.price) ? "price not listed" : esc(money(m.price, 2))) + "</span>" +
        "<span>" + (isMissing(m.average_rating) ? "unrated"
          : Number(m.average_rating).toFixed(1) + "★ (n=" + n(m.rating_number) + ")") + "</span>" +
        (m.has_review_evidence ? "<span>reviews mined</span>" : "") +
        "</div>" +
        (m.parent_asin ? '<div class="ev__actions"><button type="button" class="btn btn--sm btn--ghost" ' +
          'data-focus-asin="' + esc(m.parent_asin) + '">Open ' + esc(m.parent_asin) +
          " in the dashboard</button></div>" : "") +
        "</div>";
    }
    if (ev.kind === "review") {
      return '<div class="ev ev--review" id="' + id + '" tabindex="-1">' +
        '<div class="ev__head"><span class="ev__marker">' + esc(ev.marker) + "</span>" +
        '<span class="ev__title">verbatim review · ' + esc(aspectLabel(m.aspect || "general")) +
        "</span></div>" +
        '<div class="ev__quote">&ldquo;' + esc(m.snippet || "") + "&rdquo;</div>" +
        '<div class="ev__meta">' +
        "<span>about " + esc(truncate(m.title || "?", 60)) + " " + esc(m.product_marker || "") + "</span>" +
        (isMissing(m.rating) ? "" : "<span>" + Number(m.rating).toFixed(0) + "★</span>") +
        (isMissing(m.polarity) ? "" : "<span>polarity " + Number(m.polarity).toFixed(2) + "</span>") +
        (isMissing(m.review_year) ? "" : "<span>" + n(m.review_year) + "</span>") +
        (m.verified_purchase ? "<span>verified purchase</span>" : "") +
        (isMissing(m.helpful_vote) ? "" : "<span>" + n(m.helpful_vote) + " helpful</span>") +
        "</div></div>";
    }
    return '<div class="ev ev--stat" id="' + id + '" tabindex="-1">' +
      '<div class="ev__head"><span class="ev__marker">' + esc(ev.marker) + "</span>" +
      '<span class="ev__title">market statistic</span></div>' +
      '<div class="ev__quote">' + esc(ev.text || m.label || "") + "</div>" +
      '<div class="ev__meta"><span>source: ' + esc(m.source || "pricing.py") + "</span></div></div>";
  }

  function auditHTML(audit, msgId) {
    var bad = (audit.unsupported_markers || []).length +
      (audit.unverified_numbers || []).length +
      (audit.misattributed_reviews || []).length;
    var ok = bad === 0;
    var badge = '<button type="button" class="audit-badge audit-badge--' + (ok ? "ok" : "bad") +
      '" data-audit="' + msgId + '" aria-expanded="false">' +
      (ok ? "✓ grounded" : "⚠ " + n(bad) + " audit flag" + (bad === 1 ? "" : "s")) + "</button>";

    var detail = '<div class="audit-detail" id="audit-' + msgId + '" hidden><dl style="margin:0">' +
      "<dt>Invented citation markers (" + n((audit.unsupported_markers || []).length) + ")</dt><dd>" +
      ((audit.unsupported_markers || []).length
        ? esc(audit.unsupported_markers.join(", "))
        : "none — every marker in the answer resolves to a real evidence block") + "</dd>" +
      "<dt>Unverified numbers (" + n((audit.unverified_numbers || []).length) + ")</dt><dd>" +
      ((audit.unverified_numbers || []).length
        ? esc(audit.unverified_numbers.join(", "))
        : "none — every price, spec figure and percentage appears in the evidence") + "</dd>" +
      "<dt>Misattributed reviews (" + n((audit.misattributed_reviews || []).length) + ")</dt><dd>" +
      ((audit.misattributed_reviews || []).length
        ? "<ul>" + audit.misattributed_reviews.map(function (m) {
            return "<li>" + esc(m.review) + " belongs to " + esc(m.belongs_to) +
              " but was cited with " + esc(m.cited_with) + "</li>";
          }).join("") + "</ul>"
        : "none — no review was quoted against a product other than its own") + "</dd>" +
      "<dt>Sentences with no citation (" + n((audit.uncited_sentences || []).length) + ")</dt><dd>" +
      ((audit.uncited_sentences || []).length
        ? "<ul>" + audit.uncited_sentences.slice(0, 5).map(function (s) {
            return "<li>" + esc(truncate(s, 160)) + "</li>";
          }).join("") + "</ul>"
        : "none") + "</dd>" +
      (audit.truncated ? "<dt>Truncated</dt><dd>the generation hit its token cap</dd>" : "") +
      "<dt>How this is computed</dt><dd>" + esc(audit.explanation || "") + "</dd>" +
      "</dl></div>";

    return { badge: badge, detail: detail };
  }

  function renderAnswer(container, payload, msgId) {
    var evidence = payload.evidence || [];
    var byMarker = {};
    evidence.forEach(function (e) {
      byMarker[String(e.marker || "").replace(/[[\]]/g, "")] = e;
    });
    var audit = payload.audit || {};
    var a = auditHTML(audit, msgId);

    container.className = "msg msg--bot";
    container.innerHTML =
      '<div class="msg__bubble">' + linkifyCitations(payload.answer || "(empty answer)", msgId, byMarker) + "</div>" +
      '<div class="msg__audit">' + a.badge +
      "<span>" + n(evidence.length) + " evidence block" + (evidence.length === 1 ? "" : "s") +
      " &middot; " + pct(audit.citation_rate, 0) + " of sentences cited" +
      " &middot; " + n(payload.latency_s, 1) + " s" +
      (payload.question_type ? " &middot; " + esc(payload.question_type) : "") + "</span></div>" +
      a.detail +
      (evidence.length
        ? '<details class="evbox" id="evbox-' + msgId + '"><summary>Evidence the answer was ' +
          "generated from (" + n(evidence.length) + ")</summary>" +
          '<div class="evlist">' + evidence.map(function (e) { return evidenceHTML(e, msgId); }).join("") +
          "</div></details>"
        : '<div class="callout callout--warn"><span class="callout__icon">!</span><span>' +
          "No evidence was retrieved for this question, so the answer is not grounded in the " +
          "catalogue. Treat it as a refusal, not a result.</span></div>");

    // citation marker -> reveal its evidence block
    $$(".cite", container).forEach(function (btn) {
      btn.addEventListener("click", function () {
        var marker = btn.getAttribute("data-marker");
        var target = document.getElementById("ev-" + msgId + "-" + marker);
        if (!target) {
          toast("Citation " + marker + " matches no evidence block — the audit flagged it as invented.");
          return;
        }
        var box = document.getElementById("evbox-" + msgId);
        if (box) box.open = true;
        $$(".ev", container).forEach(function (e) { e.classList.remove("is-hit"); });
        $$(".cite", container).forEach(function (c) { c.classList.remove("is-active"); });
        btn.classList.add("is-active");
        target.classList.add("is-hit");
        target.scrollIntoView({ block: "nearest", behavior: "smooth" });
        target.focus({ preventScroll: true });
      });
    });

    var ab = $(".audit-badge", container);
    if (ab) {
      ab.addEventListener("click", function () {
        var d = document.getElementById("audit-" + msgId);
        d.hidden = !d.hidden;
        ab.setAttribute("aria-expanded", d.hidden ? "false" : "true");
      });
    }

    $$("[data-focus-asin]", container).forEach(function (b) {
      b.addEventListener("click", function () {
        setFocus(b.getAttribute("data-focus-asin"));
        activateTab("overview");
        toast("Focus product set to " + b.getAttribute("data-focus-asin"));
      });
    });

    chatScroll();
  }

  function sendQuestion(text) {
    if (!text || text.length < 3) { toast("Please ask a question of at least 3 characters."); return; }
    if (STATE.chat.busy) { toast("An answer is already being generated."); return; }

    STATE.chat.busy = true;
    var st = STATE.chat.llm || { state: "unloaded" };
    llmPillRender(st, null);
    startStatusPolling();

    appendUserMessage(text);
    var msgId = "m" + (++STATE.chat.msgSeq);
    var stopTimer = appendThinking(msgId);
    var container = document.getElementById(msgId);

    api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, max_new_tokens: 420, temperature: 0.0 })
    }).then(function (payload) {
      stopTimer();
      try {
        renderAnswer(container, payload, msgId);
      } catch (renderErr) {
        // a render crash must not swallow the answer or wedge the panel
        container.className = "msg msg--bot msg--error";
        container.innerHTML = '<div class="msg__bubble"><strong>The answer arrived but could not ' +
          "be rendered.</strong> " + esc(renderErr.message) +
          "<details style=\"margin-top:8px\"><summary style=\"cursor:pointer\">Raw answer text" +
          "</summary><div style=\"white-space:pre-wrap;margin-top:6px\">" +
          esc((payload && payload.answer) || "(none)") + "</div></details></div>";
        chatScroll();
      }
    }, function (err) {
      stopTimer();
      container.className = "msg msg--bot msg--error";
      var body = err.body || {};
      var e = body.error || {};
      var extra = "";
      if (e.code === "gpu_busy" || (body.detail && body.detail.code === "gpu_busy")) {
        extra = " Another process is holding the GPU; the server refuses to load the 7B model " +
          "rather than risk an out-of-memory crash. Try again once it frees up.";
      } else if (err.status === 429) {
        extra = " Only one answer can be generated at a time on a single GPU.";
      } else if (err.status === 0) {
        extra = " The API is unreachable — check that the server is still running.";
      }
      container.innerHTML = '<div class="msg__bubble"><strong>The answer failed' +
        (err.status ? " (HTTP " + err.status + ")" : "") + ".</strong> " +
        esc(err.message) + esc(extra) +
        '<div style="margin-top:8px"><button type="button" class="btn btn--sm" ' +
        'data-retry-q="' + esc(text) + '">Retry this question</button></div></div>';
      var rb = $("[data-retry-q]", container);
      if (rb) rb.addEventListener("click", function () { sendQuestion(text); });
      chatScroll();
    }).then(release, release);

    // released on BOTH settle paths: a throw inside either handler must never
    // leave STATE.chat.busy stuck true, which would disable the input for good
    function release() {
      STATE.chat.busy = false;
      pollChatStatus().catch(function () {
        // status unreachable: re-enable locally so the panel is not wedged
        $("#chatInput").disabled = false;
        $("#chatSend").disabled = false;
      });
    }
  }

  /* ==================================================================== */
  /* 9. Wiring                                                            */
  /* ==================================================================== */

  function wire() {
    $("#searchForm").addEventListener("submit", function (e) {
      e.preventDefault();
      runSearch(false);
    });
    $("#resetFilters").addEventListener("click", function () {
      $("#searchForm").reset();
      runSearch(false);
    });
    $("#loadMore").addEventListener("click", function () { runSearch(true); });
    ["#fSegment", "#fRam", "#fGpu", "#fSort"].forEach(function (sel) {
      $(sel).addEventListener("change", function () { runSearch(false); });
    });
    ["#fSent", "#fRenewedOut"].forEach(function (sel) {
      $(sel).addEventListener("change", function () { runSearch(false); });
    });

    $("#results").addEventListener("click", function (e) {
      var btn = e.target.closest ? e.target.closest(".result") : null;
      if (btn) setFocus(btn.getAttribute("data-asin"));
    });

    $$(".tab").forEach(function (t) {
      t.addEventListener("click", function () { activateTab(t.getAttribute("data-tab")); });
    });

    $("#healthPill").addEventListener("click", function () { loadHealth().catch(function () { }); });

    $("#chatForm").addEventListener("submit", function (e) {
      e.preventDefault();
      var input = $("#chatInput");
      var text = input.value.trim();
      if (!text) return;
      input.value = "";
      sendQuestion(text);
    });
    $("#chatInput").addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!$("#chatSend").disabled) $("#chatForm").dispatchEvent(new Event("submit", { cancelable: true }));
      }
    });
    $("#chatSuggestions").addEventListener("click", function (e) {
      var chip = e.target.closest ? e.target.closest(".chip") : null;
      if (!chip) return;
      if ($("#chatSend").disabled) { toast("The model is not accepting input yet."); return; }
      sendQuestion(chip.getAttribute("data-q"));
    });
  }

  function boot() {
    wire();
    loadFilterOptions();
    loadHealth().catch(function () { });
    loadOverview().catch(function () { });
    pollChatStatus().catch(function () { });
    runSearch(false);
    renderOverviewTab();

    // pick a sensible default focus so the dashboard is never empty on arrival
    api("/api/products/search" + qs({ sort: "reviews_desc", has_sentiment: "true", limit: 1 }))
      .then(function (d) {
        if (d.items && d.items.length && !STATE.focusAsin) setFocus(d.items[0].parent_asin);
      }, function () { /* the empty-state copy already covers this */ });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
