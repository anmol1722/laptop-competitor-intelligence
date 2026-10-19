"""Chart correctness and chat integration; synthetic ground truth, no hosted calls."""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web import api
from web.chat_controls import ChatControls
from web.chat_charts import chart_reply, wants_chart


def catalogue():
    rows=[]
    for i,(brand,segment,price,rating,ram) in enumerate([
        ('HP','business',100.,4.,8),('HP','business',300.,5.,16),
        ('HP','business',None,4.5,8),('Lenovo','business',500.,4.2,16),
        ('Lenovo','gaming',1200.,3.8,32),('Acer','gaming',None,None,4),
    ]):
        rows.append(dict(parent_asin=f'B0{i:08d}',title=f'{brand} Model {100+i}',brand=brand,
                         segment=segment,price=price,average_rating=rating,ram_gb=ram,
                         storage_gb=512,screen_in=15.6,cpu_tier=7,os_family='Windows',is_renewed=False,
                         is_discrete_gpu=segment=='gaming',rating_number=10+i,n_reviews=5+i))
    return pd.DataFrame(rows)


class ChartDataTests(unittest.TestCase):
    def setUp(self): self.products=catalogue()
    def reply(self,q,ps=None): return chart_reply(q,self.products,ps)
    def chart(self,q,ps=None): return self.reply(q,ps)['charts'][0]

    def test_brand_medians_use_all_matching_rows_and_keep_missing_coverage(self):
        c=self.chart('Graph median prices of HP and Lenovo business laptops')
        rows={r['label']:r for r in c['rows']}
        self.assertEqual(rows['HP']['value'],200)
        self.assertEqual(rows['HP']['n'],2)
        self.assertEqual(rows['HP']['total'],3)
        self.assertAlmostEqual(rows['HP']['coverage'],2/3)
        self.assertEqual(rows['Lenovo']['value'],500)
        self.assertFalse(rows['HP']['reliable'])
        self.assertEqual(c['n_total'],4)
        self.assertEqual(c['n_valid'],3)

    def test_budget_ram_and_segment_filters_are_not_relaxed(self):
        c=self.chart('Graph mean price by brand for business laptops under $400 with 16 GB RAM')
        self.assertEqual(c['n_total'],1)
        self.assertEqual(c['rows'][0]['value'],300)
        self.assertEqual(c['filters']['ram_min'],16)
        self.assertEqual(self.reply('Graph median price under $15')['charts'],[])
        self.assertEqual(self.reply('Graph median price between $5 and $10')['charts'],[])

    def test_cross_segment_comparison_does_not_infer_a_single_os(self):
        self.products.loc[5,['segment','os_family','price']]=['chromebook','ChromeOS',250]
        c=self.chart('Graph median prices by segment for gaming laptops vs chromebooks')
        self.assertEqual({r['label']:r['value'] for r in c['rows']},{'gaming':1200,'chromebook':250})

    def test_unknown_prices_are_excluded_from_distribution_not_imputed(self):
        c=self.chart('Show price distribution')
        self.assertEqual(sum(r['value'] for r in c['rows']),4)
        self.assertEqual(c['rows'][0]['low'],100)
        self.assertEqual(c['n_total'],6)
        self.assertEqual(self.reply('Graph Acer prices')['chart_status'],'unavailable')

    def test_scatter_preserves_price_rating_pair_identity_and_sampling_is_stable(self):
        c=self.chart('Plot price vs rating')
        pairs={r['asin']:(r['x'],r['value']) for r in c['rows']}
        self.assertEqual(pairs['B000000000'],(100,4))
        self.assertEqual(len(pairs),4)
        self.assertEqual(c['rows'],self.chart('Plot price vs rating')['rows'])
        self.assertEqual(c['unit'],'rating / 5')

    def test_listing_pie_is_a_complete_partition_of_filtered_catalogue(self):
        c=self.chart('Show a pie chart of laptop listings by brand')
        self.assertEqual(c['type'],'donut')
        self.assertEqual(sum(r['value'] for r in c['rows']),6)
        self.assertIn('not sales',' '.join(c['notes']))

    def test_review_counts_are_retained_reviews_not_listing_or_rating_counts(self):
        c=self.chart('Graph number of reviews by brand')
        self.assertEqual(c['unit'],'reviews')
        self.assertEqual(sum(r['value'] for r in c['rows']),45)

    def test_sentiment_is_weighted_by_review_count_and_zero_mentions_stay_missing(self):
        ps=pd.DataFrame([
            dict(parent_asin='B000000000',n_reviews_scored=10,overall_pos_share=.9,battery_mentions=10,battery_pos_share=.9,display_mentions=0,display_pos_share=np.nan),
            dict(parent_asin='B000000001',n_reviews_scored=90,overall_pos_share=.1,battery_mentions=90,battery_pos_share=.1,display_mentions=0,display_pos_share=np.nan),
        ]).set_index('parent_asin',drop=False)
        c=self.chart('Graph positive battery sentiment for HP',ps)
        self.assertEqual(len(c['rows']),1)
        self.assertEqual(c['rows'][0]['value'],18.)  # unweighted would be 50%
        self.assertEqual(c['rows'][0]['n'],100)
        self.assertEqual(self.reply('Graph positive display sentiment for HP',ps)['charts'],[])

    def test_negative_sentiment_is_not_assumed_to_be_one_minus_positive(self):
        ps=pd.DataFrame([dict(parent_asin='B000000000',n_reviews_scored=3,overall_pos_share=1/3,battery_mentions=3,battery_pos_share=1/3)])
        reviews=pd.DataFrame(dict(parent_asin=['B000000000']*3,asp_battery_pol=[1.,0.,-1.]))
        with patch('web.chat_charts.sentiment.load_review_sentiment',return_value=reviews):
            c=self.chart('Graph negative battery sentiment for HP',ps)
        self.assertAlmostEqual(c['rows'][0]['value'],100/3,places=3)
        self.assertEqual(c['polarity'],'negative')

    def test_unknown_model_time_series_and_currency_do_not_fall_back_to_wrong_scope(self):
        for q in ('Graph price of HP Model 99999','Graph prices over time','Graph battery life in hours','Graph prices in CAD',
                  'Graph prices by RAM','Show a scatter plot of battery and price'):
            with self.subTest(q=q): self.assertEqual(self.reply(q)['chart_status'],'unavailable')

    def test_replies_are_json_safe_including_all_missing_price_group(self):
        for q in ('Graph median price by brand','Plot price vs rating','Show price distribution','Graph Acer price'):
            json.dumps(self.reply(q),allow_nan=False)

    def test_ordinary_questions_and_text_only_comparisons_keep_rag(self):
        self.assertFalse(wants_chart('What is the best laptop under $1200?'))
        self.assertFalse(wants_chart('Compare HP and Lenovo prices, text only'))
        self.assertFalse(wants_chart('Compare HP and Lenovo prices, no graph'))
        self.assertIsNone(self.reply('How do I use this application?'))


class ChartAPITests(unittest.TestCase):
    def test_chart_api_works_without_key_or_provider_and_caches(self):
        products=catalogue()
        with patch.object(api.STATE,'products',products), patch.object(api,'CHAT_CONTROLS',ChatControls()), \
             patch.object(api,'CHART_RATE_LIMITER',ChatControls(limits=((20,60),))), \
             patch.object(api.LLM,'ensure_ready',side_effect=AssertionError('Chart loaded LLM')), \
             patch.object(api.LLM,'answer',side_effect=AssertionError('Chart called provider')):
            client=TestClient(api.app)
            first=client.post('/api/chat',json={'question':'Graph median price by brand'})
            second=client.post('/api/chat',json={'question':'Graph median price by brand'})
        self.assertEqual(first.status_code,200,first.text)
        body=first.json()
        self.assertEqual(body['response_mode'],'data_chart')
        self.assertEqual(body['charts'][0]['type'],'bar')
        self.assertEqual(body['charts'][0]['source'],'[S1]')
        self.assertTrue(body['audit']['grounded'])
        self.assertFalse(body['cached'])
        self.assertTrue(second.json()['cached'])
        self.assertEqual(body['charts'],second.json()['charts'])
        json.dumps(body,allow_nan=False)

    def test_chart_requests_have_separate_limits_from_hosted_text(self):
        with patch.object(api.STATE,'products',catalogue()), patch.object(api,'CHAT_CONTROLS',ChatControls(limits=((1,60),))), \
             patch.object(api,'CHART_RATE_LIMITER',ChatControls(limits=((2,60),))), \
             patch.object(api.LLM,'ensure_ready',side_effect=AssertionError('Chart called provider')):
            client=TestClient(api.app)
            for _ in range(2): self.assertEqual(client.post('/api/chat',json={'question':'Graph prices'}).status_code,200)
            limited=client.post('/api/chat',json={'question':'Graph prices'})
            self.assertEqual(limited.status_code,429)
            self.assertEqual(limited.json()['error']['code'],'client_rate_limited')

if __name__=='__main__': unittest.main()
