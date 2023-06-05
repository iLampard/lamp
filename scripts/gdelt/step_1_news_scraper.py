
import os
import pandas as pd

import feapder
from newspaper import Article


def parse_html_to_title_and_article(html_doc):
    article = Article('')  # STRING REQUIRED AS `url` ARGUMENT BUT NOT USED
    article.set_html(html_doc)
    article.parse()
    title = article.title
    content = article.text
    return title, content


class BaseSpider:
    """
    Download the news text from given urls in GDELT data
    """
    def __init__(
            self,
            db_storage: str,
            gdelt_data_fn: str,
            is_continue: bool = True,
            is_inverse: bool = False,
            thread_count: int = None,
    ):
        self.db_storage = db_storage
        self.gdelt_data_fn = gdelt_data_fn
        self.is_continue = is_continue
        self.is_inverse = is_inverse
        self._gdelt_data = None  # load data in 'start_requests'

    @property
    def gdelt_data(self):
        if self._gdelt_data is None:
            self._gdelt_data = pd.read_csv(self.gdelt_data_fn)
        return self._gdelt_data

    def _is_a_error_request(self, title, article):
        skip_titles = [
            'Just a moment',
            'Page Unavailable',
            'Page Not Found',
            'Access denied',
            'Attention Required!',
        ]

        skip_articles = [
            '请检查您的互联网连接是否正常',
            '无法访问此网站',
            'The item that you have requested was not found',
        ]

        if title is None or title.strip() == '':
            return True
        if article is None or article.strip() == '':
            return True

        for key_title in skip_titles:
            if key_title in title:
                return True

        for key_article in skip_articles:
            if key_article in article:
                return True
        return False

    def _save_news(self, gdelt_series, title, article):
        path = self._construct_news_path(gdelt_series)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'article_{title}.txt'), 'w') as file:
            file.write(article)
        return True

    def _construct_news_path(self, gdelt_series):
        return os.path.join(self.db_storage, str(gdelt_series['DATEADDED']), str(gdelt_series['GLOBALEVENTID']))

    def get_scraped_news_count(self):
        if not os.path.exists(self.db_storage):
            return 0
        count = 0
        for ts_id in os.listdir(self.db_storage):
            ts_id_path = os.path.join(self.db_storage, ts_id)
            if os.path.isdir(ts_id_path):
                for event_id in os.listdir(ts_id_path):
                    event_id_path = os.path.join(ts_id_path, event_id)

                    if os.path.isdir(event_id_path) and len(os.listdir(event_id_path)) > 0:
                        count += 1

        return count

    def get_total_news(self):
        return len(self.gdelt_data)


class NewsRender(feapder.AirSpider, BaseSpider):
    __custom_setting__ = dict(
        # SPIDER_THREAD_COUNT=32,
        SPIDER_MAX_RETRY_TIMES=1,
        REQUEST_TIMEOUT=300,

        # LOG_IS_WRITE_TO_FILE=True,
        LOG_LEVEL='ERROR',
        PRINT_EXCEPTION_DETAILS=False,

        WEBDRIVER=dict(
            pool_size=16,  # 浏览器的数量
            load_images=False,  # 是否加载图片
            # user_agent=None,  # 字符串 或 无参函数，返回值为user_agent
            user_agent="User-Agent': 'Mozilla/5.0 (iPad; U; CPU OS 3_2_1 like Mac OS X; en-us) AppleWebKit/531.21.10 (KHTML, like Gecko) Mobile/7B405",
            proxy=None,  # xxx.xxx.xxx.xxx:xxxx 或 无参函数，返回值为代理地址
            headless=False,  # 是否为无头浏览器
            driver_type="CHROME",  # CHROME、PHANTOMJS、FIREFOX
            timeout=120,  # 请求超时时间
            window_size=(1024, 800),  # 窗口大小
            executable_path=None,  # 浏览器路径，默认为默认路径
            # render_time=5,  # 渲染时长，即打开网页等待指定时间后再获取源码
            custom_argument=[
                "--ignore-certificate-errors",
                "--disable-blink-features=AutomationControlled",
            ],  # 自定义浏览器渲染参数
            xhr_url_regexes=[
                "/ad",
            ],  # 拦截 http://www.spidertools.cn/spidertools/ad 接口
            auto_install_driver=True,
        )
    )

    def __init__(
            self,
            db_storage: str,
            gdelt_data_fn: str,
            is_continue: bool = True,
            is_inverse: bool = False,
            thread_count: int = None,
    ):
        feapder.AirSpider.__init__(self, thread_count)
        BaseSpider.__init__(
            self,
            db_storage=db_storage,
            gdelt_data_fn=gdelt_data_fn,
            is_continue=is_continue,
            is_inverse=is_inverse,
            thread_count=thread_count
        )

    def start_requests(self):
        if self.is_inverse:
            it = range(len(self.gdelt_data) - 1, -1, -1)
        else:
            it = range(410000, len(self.gdelt_data))
        for idx in it:
            gdelt_series = self.gdelt_data.iloc[idx]
            path = self._construct_news_path(gdelt_series)
            if self.is_continue:
                # check if already fetched
                if os.path.exists(path) and len(os.listdir(path)) > 0:
                    continue

            url = gdelt_series['SOURCEURL']
            yield feapder.Request(
                url,
                random_user_agent=False,
                stream=False,
                render_time=5,
                render=True,
                verify=False,
                timeout=(30, 120),
                allow_redirects=True,
                extra_info={
                    'DATEADDED': gdelt_series['DATEADDED'],
                    'GLOBALEVENTID': gdelt_series['GLOBALEVENTID'],
                }
            )

    def parse(self, request, response):
        # browser: WebDriver = response.browser
        url = request.url
        title, article = parse_html_to_title_and_article(response.text)

        # check status
        if self._is_a_error_request(title, article):
            raise RuntimeError(f'Scrape fail: {url}')

        # ---------------------------------------------
        gdelt_series = request.extra_info
        # save news
        self._save_news(gdelt_series, title, article)


if __name__ == "__main__":
    gdelt_data_fn = '../../data/gdelt/events_raw_clean.csv'
    render = NewsRender(
        db_storage='../../data/gdelt/news',
        gdelt_data_fn=gdelt_data_fn,
        thread_count=24,
        # is_inverse=True
    )

    # statistics
    print('Scraped news count', render.get_scraped_news_count())
    print('Total news count:', render.get_total_news())

    # start spider
    render.start()
