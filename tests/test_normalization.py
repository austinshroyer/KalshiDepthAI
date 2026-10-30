from kalshi_quant.shared.collection.classification import Classifier, MarketClass
from kalshi_quant.shared.collection.rollups import projected
from kalshi_quant.shared.data_models.market import (
    Quote, implied_probability_e4, normalize_market, normalize_orderbook, normalize_quote, to_e2, to_e4,
)
from kalshi_quant.shared.utilities.timeutil import parse_ts_ms


def test_fixed_point_parsing_is_exact():
    assert to_e4("0.0010") == 10 and to_e4("0.5300") == 5300 and to_e4("1.0000") == 10_000
    assert to_e2("1656548.93") == 165654893 and to_e2("5.38") == 538
    assert to_e4(None) is None and to_e4("") is None


def test_empty_sides_become_none():
    assert normalize_quote(0, 0) == (None, None)
    assert normalize_quote(0, 10_000) == (None, None)
    assert normalize_quote(5300, 5400) == (5300, 5400)


def test_no_prices_are_complements():
    quote = Quote(4000, 4200)
    assert quote.no_bid == 5800 and quote.no_ask == 6000 and quote.yes_prob == 4100


def test_probability_falls_back_to_last_trade():
    assert implied_probability_e4(None, 4200, 4100) == 4100
    assert implied_probability_e4(None, None, None) is None


def test_decimal_examples_from_the_brief():
    assert Quote(5000, 5000).yes_prob / 10_000 == 0.50
    assert Quote(7500, 7500).yes_prob / 10_000 == 0.75
    assert Quote(200, 200).yes_prob / 10_000 == 0.02


def test_real_empty_market_payload():
    m = normalize_market({"ticker": "A-B-C", "event_ticker": "A-B", "status": "active", "yes_bid_dollars": "0.0000",
                          "yes_ask_dollars": "0.0000", "no_bid_dollars": "1.0000", "no_ask_dollars": "1.0000",
                          "last_price_dollars": "0.0000", "volume_fp": "0.00"})
    assert m.quote.yes_bid is None and m.quote.yes_ask is None and m.quote.last is None
    assert m.series_ticker == "A" and m.volume == 0


def test_orderbook_best_first_and_implied_ask():
    book = normalize_orderbook({"orderbook_fp": {"yes_dollars": [["0.5100", "10.00"], ["0.5300", "5.00"]],
                                                 "no_dollars": [["0.4400", "7.00"], ["0.4600", "9.00"]]}})
    assert book.yes_bid == 5300 and book.yes_ask == 5400
    assert book.no[0] == (4600, 900)


def test_timestamps():
    assert parse_ts_ms("2026-10-30T21:32:12.388381Z") == 1793395932388
    assert parse_ts_ms("2026-08-31T00:00:00.123456789+00:00") == 1788134400123
    assert parse_ts_ms("") is None and parse_ts_ms(1000) == 1_000_000


def classifier():
    c = Classifier()
    c.update_series([
        {"ticker": "KXBTC15M", "frequency": "fifteen_min"}, {"ticker": "KXGBPUSD15MTEST", "frequency": "fifteen_min"},
        {"ticker": "KXNBADRAFTTOP15", "frequency": "annual", "title": "NBA Draft Top 15 Pick"},
        {"ticker": "HOUSETX15", "frequency": "one_off", "title": "House TX 15"},
        {"ticker": "KXT20FIRST15", "frequency": "custom", "title": "T20 First 15 Overs Winner"},
        {"ticker": "KXBTCH", "frequency": "hourly"},
    ])
    return c


def test_real_series_lookalikes_stay_general():
    for ticker in ("KXNBADRAFTTOP15", "HOUSETX15", "KXT20FIRST15"):
        assert classifier().classify(ticker).market_class is MarketClass.GENERAL


def test_metadata_decides_fifteen_minute():
    c = classifier()
    assert c.classify("KXBTC15M", duration_s=900).market_class is MarketClass.FIFTEEN_MINUTE
    assert c.classify("KXBTCH", event_cadence="fifteen_min").market_class is MarketClass.FIFTEEN_MINUTE
    assert c.classify("KXBTCH").market_class is MarketClass.GENERAL


def test_exclusions():
    c = classifier()
    assert c.classify("KXGBPUSD15MTEST").market_class is MarketClass.EXCLUDED_TEST
    assert c.classify("KXMVECROSSCATEGORY").market_class is MarketClass.EXCLUDED_MULTIVARIATE
    assert c.classify("X", mve_collection="KXMVE-R").market_class is MarketClass.EXCLUDED_MULTIVARIATE
    assert c.classify("X", market_type="perpetual").market_class is MarketClass.EXCLUDED_PERPETUAL


def test_duration_mismatch_is_audited_not_reassigned():
    c = classifier()
    odd = c.classify("KXBTC15M", duration_s=3600)
    assert odd.market_class is MarketClass.FIFTEEN_MINUTE and "3600" in odd.anomaly
    sneaky = c.classify("KXBTCH", duration_s=900)
    assert sneaky.market_class is MarketClass.GENERAL and "900s" in sneaky.anomaly


def test_projected_winner():
    assert projected(7000) == ("yes", 7000)
    assert projected(3000) == ("no", 7000)
    assert projected(5000) == ("tie", 5000)
    assert projected(None) == (None, None)
