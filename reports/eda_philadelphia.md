# EDA — Philadelphia

## Overview (table row counts)

| table      |    rows |
|:-----------|--------:|
| businesses |   44840 |
| reviews    | 1930159 |
| users      |  513947 |
| tips       |  235828 |
| checkins   |   38781 |

## Businesses by state

| state   |   businesses |   yelp_review_count |
|:--------|-------------:|--------------------:|
| PA      |        34039 |         1.54079e+06 |
| NJ      |         8536 |    249837           |
| DE      |         2265 |     67370           |

## Top cities

| city            | state   |   businesses |
|:----------------|:--------|-------------:|
| Philadelphia    | PA      |        14567 |
| Wilmington      | DE      |         1445 |
| Cherry Hill     | NJ      |          959 |
| West Chester    | PA      |          838 |
| King of Prussia | PA      |          560 |
| Doylestown      | PA      |          539 |
| Bensalem        | PA      |          454 |
| Norristown      | PA      |          448 |
| Exton           | PA      |          419 |
| Marlton         | NJ      |          415 |
| Lansdale        | PA      |          378 |
| Ardmore         | PA      |          376 |
| Wayne           | PA      |          375 |
| Media           | PA      |          371 |
| Phoenixville    | PA      |          365 |

## Review date range

| first_review        | last_review         |   n_reviews |
|:--------------------|:--------------------|------------:|
| 2005-02-16 04:06:26 | 2022-01-19 19:48:45 |     1930159 |

## Reviews per user (within metro)

|   n_users |   min |   p50 |   p90 |   p99 |   max |    mean |   one_review_users |
|----------:|------:|------:|------:|------:|------:|--------:|-------------------:|
|    513954 |     1 |     1 |     7 |    39 |  3046 | 3.75551 |             290208 |

## Reviews per business

|   n_businesses |   min |   p50 |   p90 |    p99 |   max |    mean |
|---------------:|------:|------:|------:|-------:|------:|--------:|
|          44840 |     5 |    15 |    95 | 435.61 |  5778 | 43.0455 |

## Rating distribution

|   stars |      n |   pct |
|--------:|-------:|------:|
|       1 | 306710 |  15.9 |
|       2 | 163638 |   8.5 |
|       3 | 208328 |  10.8 |
|       4 | 427541 |  22.2 |
|       5 | 823942 |  42.7 |

## Reviews per year

|   yr |      n |
|-----:|-------:|
| 2005 |    151 |
| 2006 |   1257 |
| 2007 |   7518 |
| 2008 |  20902 |
| 2009 |  35094 |
| 2010 |  56253 |
| 2011 |  82109 |
| 2012 | 101424 |
| 2013 | 129616 |
| 2014 | 159621 |
| 2015 | 194861 |
| 2016 | 205333 |
| 2017 | 219684 |
| 2018 | 226241 |
| 2019 | 223942 |
| 2020 | 126928 |
| 2021 | 132545 |
| 2022 |   6680 |

## Review length (chars)

|   min |   p50 |   p90 |   p99 |   max |    mean |
|------:|------:|------:|------:|------:|--------:|
|     1 |   429 |  1217 |  2680 |  5000 | 590.298 |

## Account age vs lifetime reviews

| account_age   |   n_users |   avg_lifetime_reviews |   median_lifetime_reviews |
|:--------------|----------:|-----------------------:|--------------------------:|
| 0  <30d       |       999 |                1.51451 |                         1 |
| 1  30-180d    |      3680 |                1.94946 |                         1 |
| 2  180-365d   |      5013 |                3.22701 |                         1 |
| 3  1-3y       |     34657 |                4.85971 |                         2 |
| 4  >3y        |    469598 |               29.5143  |                         5 |

## Figures

![rating_distribution](figures/rating_distribution.png)
![reviews_per_year](figures/reviews_per_year.png)
![reviews_per_user](figures/reviews_per_user.png)
