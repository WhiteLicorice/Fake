from __future__ import annotations

import contextlib
import csv
import io
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parent
SERVER_DIR = REPO_DIR / "server"
MODEL_PATH = SERVER_DIR / "root" / "models" / "LogisticRegression.pkl"
REPORT_PATH = REPO_DIR / "new_fp_article.md"
LOG_PATH = REPO_DIR / "compose_fp_article.log"


FN_TEXT = (
    "Binigyan ng tatlong taong extension para sa kanyang panunungkulan "
    "bilang Commissioner ng Philippine Basketball Association si Willie "
    "Marcial. Sa kanilang annual planning session sa Star Hotels sa "
    "bansang Italya, gaya ng naging pagkakatalaga sa kanya bilang "
    "Commissioner ng liga, naging unanimous ang pagbibigay ng board of "
    "governors ng extension sa termino ni Marcial kahapon (Huwebes). May "
    "nalalabi pang isang taon sa naunang tatlong taong kontrata na "
    "nilagdaan ni Marcial noong 2018 pero binigyan sya ng PBA board ng "
    "bagong vote of confidence. Ito'y bunga na rin ng magandang "
    "performance nito na nagustuhan ng board. We're open and very "
    "transparent about his performance, wika ni PBA Chairman Ricky Vargas "
    "tungkol kay Marcial."
)


@dataclass(frozen=True)
class Candidate:
    article_id: str
    title: str
    text: str
    translation: str = ""


CANDIDATES = [
    Candidate(
        "cand_01",
        "DOH free barangay checkups",
        (
            "MAYNILA - Inihayag umano ng isang source sa Department of Health na "
            "pansamantalang ipapatupad ang libreng checkup program sa lahat ng "
            "barangay health center simula Lunes. Ayon sa kumalat na abiso, "
            "maaaring magtungo ang mga residente kahit walang appointment basta "
            "magdala ng valid ID at kopya ng vaccination card. Sinabi rin sa post "
            "na prayoridad ang mga senior citizen, buntis, at estudyante na may "
            "lagnat o ubo. Nilinaw umano ng source na bahagi ito ng bagong "
            "kampanya laban sa mga sakit na nauuso tuwing tag-ulan. Wala pang "
            "opisyal na memorandum na inilalabas sa website ng ahensiya, ngunit "
            "mabilis nang ibinahagi ang anunsiyo sa ilang community group. Ilang "
            "health worker ang nagsabing wala silang natatanggap na kautusan "
            "tungkol dito."
        ),
        (
            "MANILA - A source from the Department of Health allegedly announced "
            "that a temporary free checkup program would be implemented in all "
            "barangay health centers starting Monday. According to the circulating "
            "notice, residents could go even without an appointment as long as "
            "they brought a valid ID and a copy of their vaccination card. The post "
            "also said that senior citizens, pregnant women, and students with "
            "fever or cough would be prioritized. The source allegedly clarified "
            "that this was part of a new campaign against illnesses common during "
            "the rainy season. No official memorandum had been posted on the "
            "agency website, but the announcement had already been shared quickly "
            "in community groups. Several health workers said they had received no "
            "order about it."
        ),
    ),
    Candidate(
        "cand_02",
        "DOTr student free rides",
        (
            "QUEZON CITY - Naglabas umano ng bagong direktiba ang Department of "
            "Transportation na magbibigay ng libreng sakay sa mga estudyante sa "
            "lahat ng tren tuwing Biyernes. Batay sa kumalat na kopya ng advisory, "
            "kailangan lamang ipakita ang school ID sa ticket booth upang makakuha "
            "ng one-way pass. Sinabi ng post na layunin ng programa na bawasan ang "
            "gastos ng mga pamilya habang tumataas ang presyo ng pamasahe. Ayon sa "
            "isang source sa ahensiya, kasama raw sa unang yugto ang LRT, MRT, at "
            "ilang provincial bus terminal. Wala pang pahayag ang DOTr sa opisyal "
            "nitong mga channel, ngunit maraming magulang na ang nagtanong sa mga "
            "istasyon. Pinaalalahanan ng ilang operator ang publiko na hintayin muna "
            "ang kumpirmasyon bago bumiyahe."
        ),
    ),
    Candidate(
        "cand_03",
        "Fake medicine warehouse",
        (
            "CEBU CITY - Arestado umano ang isang barangay kagawad matapos mahulihan "
            "ng kahon-kahong pekeng gamot sa isang bodega malapit sa palengke. Ayon "
            "sa ulat na kumalat sa social media, nagsagawa ng operasyon ang PNP at "
            "Food and Drug Administration matapos makatanggap ng reklamo mula sa mga "
            "residente. Natagpuan daw sa lugar ang mga kapsulang may tatak ng "
            "kilalang pain reliever ngunit walang batch number. Sinabi ng isang "
            "hindi pinangalanang opisyal na ibinebenta ang mga gamot sa mas mababang "
            "presyo sa online marketplace. Itinanggi ng kampo ng kagawad ang "
            "paratang at sinabing ginagamit lamang ang bodega bilang imbakan ng "
            "relief goods. Wala pang inilalabas na opisyal na blotter ang lokal na "
            "pulisya tungkol sa insidente."
        ),
    ),
    Candidate(
        "cand_04",
        "Four-day workweek order",
        (
            "MANILA - Source ng GMA News Online ang nagsabing pinag-aaralan na ng "
            "Palasyo ang panukalang four-day workweek para sa lahat ng tanggapan ng "
            "gobyerno simula sa susunod na buwan. Ayon sa ulat, below target pa rin "
            "ang pagtitipid sa kuryente kaya nais ng economic cluster na paikliin "
            "ang araw ng pasok ngunit pahabain ang oras ng trabaho. Sinabi ng source "
            "na sakop ang national agencies, GOCCs at state universities, habang may "
            "hiwalay na patakaran para sa ospital at frontline offices. Wala pang "
            "inilalabas na memorandum ang Office of the President, ngunit ilang "
            "empleyado na raw ang nakatanggap ng paunang abiso mula sa kanilang HR "
            "units. Pinayuhan ang publiko na antabayanan ang opisyal na anunsyo sa "
            "loob ng linggo."
        ),
    ),
    Candidate(
        "cand_05",
        "Celebrity scholarship page",
        (
            "ILOILO CITY - Kumalat ang balitang may bagong scholarship fund ang "
            "isang kilalang aktres para sa mga estudyanteng anak ng magsasaka. Ayon "
            "sa post ng isang fan page, nasa P25,000 bawat semestre ang tulong at "
            "awtomatikong matatanggap ng unang 2,000 aplikante. Sinabi pa ng pahina "
            "na kailangan lamang magpadala ng pangalan, address, at larawan ng "
            "school ID sa pribadong mensahe. Maraming netizen ang nagpasa ng "
            "impormasyon sa mga group chat dahil ginamit sa anunsiyo ang larawan ng "
            "aktres sa isang lumang outreach program. Wala namang nakitang opisyal "
            "na pahayag sa kanyang verified account o sa foundation na madalas "
            "niyang katuwang. Nagbabala ang ilang guro na maaaring makuha ang "
            "personal na datos ng mga estudyante."
        ),
    ),
    Candidate(
        "cand_06",
        "Barangay fuel subsidy",
        (
            "DAVAO CITY - Nag-viral sa ilang barangay group ang umano'y bagong "
            "pondo ng DILG para sa fuel subsidy ng mga tricycle driver sa lungsod. "
            "Batay sa ipinaskil na listahan, tig-P3,000 ang ibibigay sa mga may "
            "prangkisa at rehistradong TODA member bago matapos ang buwan. Sinabi "
            "ng post na ang ayuda ay makukuha sa barangay hall matapos punan ang "
            "online form at mag-upload ng litrato ng OR/CR. Ayon sa isang source sa "
            "lokal na pamahalaan, pinag-uusapan pa lamang ang panukala at wala pang "
            "aprubadong pondo. Gayunman, marami nang driver ang pumila upang "
            "magtanong tungkol sa requirements. Hindi rin makita sa opisyal na "
            "website ng lungsod ang naturang programa."
        ),
    ),
    Candidate(
        "cand_07",
        "Mayor resignation rumor",
        (
            "BACOLOD CITY - Isang post ang nagsabing nagbitiw na umano sa tungkulin "
            "ang alkalde ng lungsod dahil sa hindi pa inilalabas na ulat ng COA. "
            "Ayon sa kumalat na screenshot, ipinadala raw ang resignation letter sa "
            "Department of the Interior and Local Government noong Biyernes ng gabi. "
            "Sinabi rin sa post na pansamantalang hahawak sa city hall ang bise "
            "alkalde habang iniimbestigahan ang ilang procurement project. Wala "
            "namang opisyal na dokumentong ipinakita maliban sa larawang malabo ang "
            "letterhead at pirma. Sa panayam sa lokal na radyo, sinabi ng tagapagsalita "
            "ng lungsod na nasa regular meeting pa ang alkalde kinabukasan. Patuloy "
            "pa ring ibinabahagi ang balita sa mga political page."
        ),
    ),
    Candidate(
        "cand_08",
        "Rice price freeze",
        (
            "MAYNILA - May kumakalat na anunsiyo na magpapatupad umano ang Department "
            "of Agriculture ng nationwide price freeze sa bigas simula sa unang araw "
            "ng susunod na buwan. Ayon sa post, P38 kada kilo ang magiging pinakamataas "
            "na presyo ng regular milled rice sa lahat ng palengke at supermarket. "
            "Sinabi rin nito na pagmumultahin ang mga tindahang lalampas sa itinakdang "
            "presyo kahit walang hiwalay na executive order. Ilang consumer group ang "
            "natuwa sa balita, ngunit may mga retailer na nagsabing wala silang "
            "natatanggap na kopya ng direktiba. Batay sa pagtingin sa official page "
            "ng ahensiya, ang huling advisory ay tungkol lamang sa monitoring ng "
            "supply. Nanawagan ang mga lokal na opisyal na huwag munang mag-panic buying."
        ),
    ),
    Candidate(
        "cand_09",
        "Hospital celebrity bill",
        (
            "TACLOBAN CITY - Isang viral post ang nagsabing babayaran umano ng isang "
            "sikat na mang-aawit ang hospital bill ng unang 500 pasyente sa pampublikong "
            "ospital sa rehiyon. Ayon sa anunsiyo, kailangan lamang magpakita ang "
            "pasyente ng billing statement at mag-register sa link na nakalagay sa "
            "caption. Ginamit sa post ang litrato ng singer sa isang relief concert "
            "noong nakaraang taon kaya maraming tagahanga ang naniwala. Sinabi ng "
            "isang staff ng ospital na walang ganitong coordination na natanggap ang "
            "kanilang billing section. Wala ring kaparehong pahayag sa opisyal na "
            "account ng artista. Ilang pamilya ang pumunta pa rin sa ospital upang "
            "magtanong, dahilan para maglabas ng paalala ang information desk."
        ),
    ),
    Candidate(
        "cand_10",
        "Helmet fine holiday",
        (
            "GENERAL SANTOS CITY - Kumalat sa mga rider group ang balitang hindi muna "
            "manghuhuli ang PNP at traffic office ng mga motoristang walang helmet "
            "sa loob ng tatlong araw. Ayon sa post, bahagi raw ito ng dry run para "
            "sa bagong road safety ordinance na magbibigay muna ng warning bago "
            "ticket. Sinabi rin ng nagpakalat na page na galing ang impormasyon sa "
            "isang meeting ng city council at mga transport leader. Ilang rider ang "
            "nagkomento na makatutulong ito sa mga bibili pa lamang ng bagong helmet. "
            "Ngunit itinanggi ng traffic enforcer sa checkpoint na may natanggap silang "
            "kautusan. Patuloy pa rin ang normal na pagpapatupad ng batas sa mga "
            "pangunahing kalsada ng lungsod."
        ),
    ),
    Candidate(
        "cand_11",
        "DepEd Saturday classes",
        (
            "PASIG CITY - May kumakalat na memorandum na nag-uutos umano sa lahat ng "
            "public school na magdaos ng regular classes tuwing Sabado upang makahabol "
            "sa learning competencies. Nakasaad sa larawan ng memo na magsisimula ang "
            "schedule sa susunod na linggo at sasaklaw sa Grade 4 hanggang Grade 12. "
            "Ayon sa post, may dagdag na service credit para sa mga guro at libreng "
            "pagkain para sa mga mag-aaral. Maraming magulang ang nagtanong sa group "
            "chat ng klase dahil mukhang opisyal ang format ng dokumento. Sinabi ng "
            "isang district supervisor na wala pang ganitong kautusan mula sa central "
            "office. Pinaalalahanan ang mga paaralan na ang pagbabago sa school calendar "
            "ay kailangang ilabas sa official channels."
        ),
    ),
    Candidate(
        "cand_12",
        "Bank account freeze rumor",
        (
            "MANDALUYONG CITY - Nagbabala ang isang viral post na ifi-freeze umano "
            "ng malalaking bangko ang savings account na walang transaction sa loob "
            "ng tatlong buwan. Batay sa ipinakitang advisory, kailangang magbayad ang "
            "depositor ng P500 activation fee upang hindi mapasama sa listahan. Sinabi "
            "ng post na inaprubahan na raw ito ng Bangko Sentral bilang hakbang laban "
            "sa dormant account fraud. Maraming netizen ang nagtanong sa customer "
            "service page ng kani-kanilang bangko dahil sa takot na mawala ang ipon. "
            "Wala namang kaparehong abiso sa website ng BSP o sa official account ng "
            "mga bangko. Ayon sa isang banker, kahina-hinala ang link na kasama sa "
            "viral message."
        ),
    ),
    Candidate(
        "cand_13",
        "Tourism senior travel pass",
        (
            "BAGUIO CITY - Ibinahagi sa mga travel group ang umano'y bagong senior "
            "citizen tourism pass na magbibigay ng libreng entrance sa lahat ng "
            "museo, park, at heritage site sa bansa. Ayon sa post, inilunsad ito ng "
            "Department of Tourism bilang bahagi ng domestic travel recovery program. "
            "Kailangan lamang daw mag-upload ng senior citizen ID at selfie sa online "
            "portal upang makakuha ng digital pass. Sinabi rin sa anunsiyo na may "
            "kasamang discount sa ilang hotel kapag ipinakita ang QR code. Ilang tour "
            "operator ang nagsabing wala silang natatanggap na advisory tungkol sa "
            "programa. Sa opisyal na website ng DOT, ang nakalistang promosyon ay "
            "para lamang sa accredited events at hindi sa libreng entrance."
        ),
    ),
    Candidate(
        "cand_14",
        "Water service waiver",
        (
            "CALOOCAN CITY - May kumalat na advisory na awtomatikong babawasan umano "
            "ng P700 ang water bill ng mga residenteng naapektuhan ng sunod-sunod na "
            "service interruption. Ayon sa post, kasunduan daw ito ng water concessionaire "
            "at city hall matapos ang mga reklamo sa mababang pressure. Sinabi rin "
            "dito na hindi na kailangang mag-file ng complaint dahil direktang lalabas "
            "ang adjustment sa susunod na billing cycle. Maraming homeowner association "
            "ang nagbahagi ng larawan ng advisory na may logo ng kumpanya. Nang "
            "tawagan ng ilang residente ang hotline, sinabing walang blanket rebate "
            "na inaprubahan at case-to-case pa rin ang review. Wala ring katulad na "
            "notice sa official website ng kumpanya."
        ),
    ),
    Candidate(
        "cand_15",
        "PAGASA signal hoax",
        (
            "LEGAZPI CITY - Nag-viral ang isang weather update na nagsasabing itinaas "
            "na umano sa Signal No. 2 ang buong Bicol dahil sa paparating na bagyo. "
            "Ayon sa graphic, inaasahang tatama ang sama ng panahon sa loob ng 24 oras "
            "kaya pinayuhan ang mga residente na mag-imbak ng pagkain at tubig. Ginamit "
            "sa larawan ang kulay at font na kahawig ng karaniwang bulletin ng PAGASA. "
            "Ilang paaralan ang nakatanggap ng tanong mula sa mga magulang kung awtomatikong "
            "suspindido na ang klase. Sa huling opisyal na bulletin, low pressure area pa "
            "lamang ang minomonitor at wala pang tropical cyclone wind signal. Pinaalalahanan "
            "ang publiko na tingnan ang timestamp ng mga weather graphic bago magbahagi."
        ),
    ),
    Candidate(
        "cand_16",
        "White van kidnapping report",
        (
            "SAN PEDRO CITY - Mabilis na kumalat sa community pages ang kuwento tungkol "
            "sa puting van na nangunguha raw ng mga bata malapit sa tatlong paaralan. "
            "Ayon sa post, nakita umano ng isang security guard ang dalawang lalaking "
            "nag-aalok ng kendi sa mga estudyante bago sumakay sa sasakyan na walang "
            "plaka. Sinabi rin ng nag-post na may blotter na sa presinto ngunit hindi "
            "pa inilalabas upang hindi magdulot ng gulo. Maraming magulang ang nagsundo "
            "nang mas maaga sa kanilang mga anak matapos mabasa ang babala. Nang tanungin "
            "ang lokal na pulisya, sinabi nilang wala pang kumpirmadong ulat o nawawalang "
            "bata kaugnay ng van. Nagpatrolya pa rin ang barangay tanod sa paligid ng "
            "mga paaralan."
        ),
    ),
    Candidate(
        "cand_17",
        "Herbal tea diabetes claim",
        (
            "CAGAYAN DE ORO CITY - Isang health page ang nagpakalat ng artikulong nagsasabing "
            "may bagong herbal tea na maaaring magpababa ng blood sugar sa loob lamang "
            "ng pitong araw. Ayon sa post, inirekomenda raw ito ng ilang doktor sa isang "
            "seminar at ligtas inumin kahit kasabay ng maintenance medicine. Sinabi rin "
            "na nagbibigay ang distributor ng libreng sample sa unang 1,000 magpapadala "
            "ng pangalan at numero ng telepono. Maraming nakatatanda ang nagtanong sa "
            "botika kung mabibili na ang produkto. Wala namang nakalistang registration "
            "number sa Food and Drug Administration database para sa naturang brand. "
            "Nagpaalala ang ilang health worker na hindi dapat palitan ng tsaa ang gamot "
            "na nireseta ng doktor."
        ),
    ),
    Candidate(
        "cand_18",
        "City curfew return",
        (
            "ANGELES CITY - Kumalat sa mga neighborhood chat ang umano'y pagbabalik ng "
            "citywide curfew mula alas-diyes ng gabi hanggang alas-kwatro ng umaga. "
            "Batay sa post, ipatutupad ito ng barangay at pulisya dahil sa pagdami raw "
            "ng kaso ng akyat-bahay sa ilang subdivision. Sinabi rin sa abiso na huhulihin "
            "ang menor de edad na nasa labas kahit may kasamang magulang, maliban kung "
            "may emergency. Ilang tindahan ang nagtanong kung kailangan nilang magsara "
            "nang mas maaga. Ayon sa opisina ng city information, wala pang ordinansa o "
            "executive order na nagbabalik ng curfew. May karagdagang patrol lamang sa "
            "mga lugar na maraming reklamo, batay sa pahayag ng lokal na pulisya."
        ),
    ),
    Candidate(
        "cand_19",
        "Actor network transfer",
        (
            "QUEZON CITY - Isang entertainment blog ang nag-ulat na lilipat umano sa "
            "ibang network ang isang sikat na aktor matapos hindi mapagkasunduan ang "
            "kontrata sa kanyang kasalukuyang management. Ayon sa artikulo, may closed-door "
            "meeting na naganap sa isang hotel at nakatakda raw ang contract signing sa "
            "susunod na linggo. Sinabi pa ng blog na kumpirmado ito ng isang source mula "
            "sa production staff, bagaman walang pinangalanan. Nag-trending agad ang "
            "pangalan ng aktor dahil inakala ng fans na kanselado na ang kanyang serye. "
            "Wala namang pahayag ang talent agency o ang network tungkol sa isyu. Ilang "
            "entertainment reporter ang nagsabing promotional shoot lamang ang dinaluhan "
            "ng aktor noong araw na binanggit sa blog."
        ),
    ),
    Candidate(
        "cand_20",
        "Voter registration extension",
        (
            "MALOLOS CITY - May kumakalat na balita na palalawigin umano ng Commission "
            "on Elections ang voter registration ng dalawang buwan sa lahat ng probinsya. "
            "Ayon sa ipinaskil na advisory, maraming aplikante ang hindi nakapagparehistro "
            "dahil sa brownout at bagyo kaya nagdesisyon ang central office na magdagdag "
            "ng schedule. Sinabi rin sa post na bukas ang mga satellite office kahit "
            "Linggo at holiday, basta may dalang birth certificate at barangay clearance. "
            "Ilang kabataan ang nagpunta sa munisipyo upang magtanong tungkol sa bagong "
            "deadline. Wala namang ganitong anunsiyo sa official Comelec page at nananatili "
            "ang dating petsa sa calendar ng opisina. Pinayuhan ang publiko na huwag "
            "umasa sa screenshot na walang control number."
        ),
    ),
    Candidate(
        "cand_21",
        "Food aid cash transfer",
        (
            "ZAMBOANGA CITY - Nag-ikot sa Facebook Messenger ang mensahe tungkol sa "
            "umano'y P5,000 food assistance para sa bawat pamilyang may anak na nag-aaral "
            "sa public school. Ayon sa mensahe, pinondohan daw ito ng DSWD at Department "
            "of Education bilang tulong sa baon at grocery ng mga estudyante. Kailangan "
            "lamang punan ang form na humihingi ng buong pangalan, birthday, school, at "
            "mobile wallet number. Maraming magulang ang nagpasa ng link sa class group "
            "chat dahil may logo ng dalawang ahensiya ang larawan. Sinabi ng lokal na "
            "social welfare office na walang ganitong payout sa kanilang schedule. Ang "
            "opisyal na ayuda, ayon sa kanila, ay dumadaan pa rin sa validated list ng "
            "beneficiaries."
        ),
    ),
    Candidate(
        "cand_22",
        "Airport terminal fee refund",
        (
            "PASAY CITY - Kumalat sa travel forums ang balitang maaari umanong mag-refund "
            "ng terminal fee ang lahat ng pasaherong bumiyahe sa NAIA sa nakaraang anim "
            "na buwan. Ayon sa post, kailangan lamang ilagay sa online form ang ticket "
            "number at bank account upang makuha ang P750 refund sa loob ng tatlong araw. "
            "Sinabi ng nagbahagi na bahagi ito ng bagong passenger service audit ng airport "
            "management. Ilang biyahero ang nagtanong sa airline counters tungkol sa form, "
            "ngunit walang staff ang nakumpirma ang programa. Wala ring advisory sa official "
            "channels ng airport authority. Pinayuhan ng mga airline ang publiko na huwag "
            "magbigay ng bank details sa hindi kilalang website."
        ),
    ),
    Candidate(
        "cand_23",
        "SIM card revalidation app",
        (
            "MAKATI CITY - Isang viral announcement ang nagsabing kailangang muling "
            "i-validate ng lahat ng mobile subscriber ang kanilang SIM card sa loob ng "
            "pitong araw gamit ang bagong government app. Ayon sa post, awtomatikong "
            "madi-deactivate ang numerong hindi makakapag-upload ng selfie, valid ID, "
            "at proof of billing bago ang deadline. Sinabi pa nito na inaprubahan ang "
            "kautusan upang mabawasan ang scam text matapos ang serye ng reklamo. Maraming "
            "subscriber ang nag-download ng link na kasama sa post kahit hindi ito nasa "
            "opisyal na app store. Wala namang abiso ang DICT o mga telco tungkol sa "
            "bagong revalidation. Nagbabala ang cybersecurity group na maaaring phishing "
            "ang naturang link."
        ),
    ),
    Candidate(
        "cand_24",
        "Checkpoint QR pass",
        (
            "BATANGAS CITY - May kumalat na notice na kailangan na umano ng QR pass ang "
            "mga motorista bago makadaan sa provincial checkpoints simula ngayong linggo. "
            "Ayon sa post, bahagi ito ng kampanya ng PNP laban sa car theft at illegal "
            "transport of goods. Kailangan daw mag-register ng plate number, lisensya, "
            "at address sa online portal upang maiwasan ang mahabang inspeksiyon. Ilang "
            "delivery rider ang nagtanong sa checkpoint kung hahanapan sila ng code bago "
            "makapasok sa expressway. Sinabi ng pulisya na regular checkpoint lamang ang "
            "ipinatutupad at wala silang QR pass system. Wala ring memorandum ang provincial "
            "government tungkol sa bagong requirement para sa mga biyahero."
        ),
    ),
    Candidate(
        "cand_25",
        "Automatic class suspension",
        (
            "NAGA CITY - Kumalat sa mga parent group ang balitang automatic nang suspended "
            "ang klase sa lahat ng antas kapag umabot sa 35 degrees Celsius ang heat index "
            "sa umaga. Ayon sa post, inilabas daw ito ng Department of Education at DOH "
            "bilang bagong patakaran laban sa heat exhaustion. Sinabi rin na hindi na "
            "kailangang hintayin ang anunsiyo ng mayor kung lalabas sa weather app ang "
            "nasabing temperatura. Ilang paaralan ang nakatanggap ng tawag mula sa mga "
            "magulang na nagtatanong kung pauuwiin na ang mga bata. Wala namang national "
            "order na ganito sa opisyal na mga channel ng DepEd. Ang local government pa "
            "rin ang naglalabas ng suspensyon batay sa aktuwal na kondisyon."
        ),
    ),
]


TRAD_TRANSFORM_ORDER = [
    "word_count",
    "sentence_count",
    "polysyll_count",
    "ave_word_length",
    "ave_phrase_count",
    "ave_syllable_count_of_word",
    "word_count_per_sentence",
]

SYLL_TRANSFORM_ORDER = [
    "consonant_cluster",
    "v_density",
    "cv_density",
    "vc_density",
    "cvc_density",
    "vcc_density",
    "cvcc_density",
    "ccvcc_density",
    "ccvccc_density",
]

TABLE13_ORDER = [
    "ave_phrase_count",
    "ave_word_length",
    "word_count_per_sentence",
    "polysyll_count",
    "word_count",
    "sentence_count",
    "ave_syllable_count_of_word",
    "cvc_density",
    "consonant_cluster",
    "cvcc_density",
    "vcc_density",
    "vc_density",
    "v_density",
    "cv_density",
    "ccvcc_density",
    "ccvccc_density",
    "readability_score",
    "count_oov_words",
    "count_stopwords",
]

DISPLAY_NAMES = {
    "ave_phrase_count": "ave-phrase-count",
    "ave_word_length": "ave-word-length",
    "word_count_per_sentence": "word-count-per-sentence",
    "polysyll_count": "polysyll-count",
    "word_count": "word-count",
    "sentence_count": "sentence-count",
    "ave_syllable_count_of_word": "ave-syllable-count-of-word",
    "cvc_density": "cvc-density",
    "consonant_cluster": "consonant-cluster",
    "cvcc_density": "cvcc-density",
    "vcc_density": "vcc-density",
    "vc_density": "vc-density",
    "v_density": "v-density",
    "cv_density": "cv-density",
    "ccvcc_density": "ccvcc-density",
    "ccvccc_density": "ccvccc-density",
    "readability_score": "readability-score",
    "count_oov_words": "count-oov-words",
    "count_stopwords": "count-stopwords",
}


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def word_count(text: str) -> int:
    return len(text.split())


def load_pipeline():
    os.chdir(SERVER_DIR)
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def read_dataset_articles() -> set[str]:
    dataset_paths = [
        REPO_DIR / "datasets" / "FakeNewsFilipino.csv",
        REPO_DIR / "datasets" / "FakeNewsFilipino2024.csv",
        REPO_DIR / "training" / "root" / "datasets" / "Cruz" / "FakeNewsFilipino_Cruz2020.csv",
        REPO_DIR / "training" / "root" / "datasets" / "Lupac" / "FakeNewsPhilippines2024_Lupac.csv",
    ]
    article_texts: set[str] = set()
    for path in dataset_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                for column in ("article", "text", "news", "news_body", "content"):
                    value = row.get(column)
                    if value:
                        article_texts.add(normalize(value))
                        break
    return article_texts


def predict_article(pipeline, text: str) -> dict[str, object]:
    predicted = int(pipeline.predict([text])[0])
    probabilities = pipeline.predict_proba([text])[0]
    classes = list(pipeline.named_steps["classifier"].classes_)
    fake_probability = float(probabilities[classes.index(0)])
    real_probability = float(probabilities[classes.index(1)])
    return {
        "predicted": predicted,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
    }


def as_float_row(values) -> list[float]:
    array = np.asarray(values, dtype=float)
    return [float(value) for value in array.reshape(array.shape[0], -1)[0]]


def extract_linguistic_features(pipeline, text: str) -> dict[str, float]:
    transformers = dict(pipeline.named_steps["features"].transformer_list)
    features: dict[str, float] = {}

    features["readability_score"] = as_float_row(transformers["read"].transform([text]))[0]
    features["count_oov_words"] = as_float_row(transformers["oov"].transform([text]))[0]
    features["count_stopwords"] = as_float_row(transformers["sw"].transform([text]))[0]

    for name, value in zip(TRAD_TRANSFORM_ORDER, as_float_row(transformers["trad"].transform([text]))):
        features[name] = value
    for name, value in zip(SYLL_TRANSFORM_ORDER, as_float_row(transformers["syll"].transform([text]))):
        features[name] = value

    return features


def deployment_feature_names(pipeline) -> tuple[list[str], int]:
    names: list[str] = []
    vectorizer_count = 0
    for name, transformer in pipeline.named_steps["features"].transformer_list:
        if name == "tfidf":
            values = [f"tfidf__{feature}" for feature in transformer.get_feature_names_out()]
            names.extend(values)
            vectorizer_count += len(values)
        elif name == "bow":
            values = [f"bow__{feature}" for feature in transformer.get_feature_names_out()]
            names.extend(values)
            vectorizer_count += len(values)
        elif name == "read":
            names.append("read__readability_score")
        elif name == "oov":
            names.append("oov__count_oov_words")
        elif name == "sw":
            names.append("sw__count_stopwords")
        elif name == "trad":
            names.extend(f"trad__{feature}" for feature in TRAD_TRANSFORM_ORDER)
        elif name == "syll":
            names.extend(f"syll__{feature}" for feature in SYLL_TRANSFORM_ORDER)
        else:
            raise ValueError(f"Unhandled transformer: {name}")

    coefficient_count = len(pipeline.named_steps["classifier"].coef_[0])
    if len(names) != coefficient_count:
        raise ValueError(
            f"Feature-name count mismatch: built {len(names)} names for "
            f"{coefficient_count} coefficients"
        )
    return names, vectorizer_count


def manuscript_feature_name(feature: str) -> str:
    if feature.startswith("bow__"):
        return "vectorizers--bow--" + feature.removeprefix("bow__")
    if feature.startswith("tfidf__"):
        return "vectorizers--tfidf--" + feature.removeprefix("tfidf__")
    return feature.replace("__", "--")


def active_vectorizer_predictors(pipeline, text: str, limit_per_direction: int = 2) -> list[dict[str, object]]:
    feature_names, vectorizer_count = deployment_feature_names(pipeline)
    feature_union = pipeline.named_steps["features"]
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    transformed = feature_union.transform([text])

    if hasattr(transformed, "tocsr"):
        row = transformed.tocsr()[0]
        indices = row.indices
        values = row.data
    else:
        dense_row = np.asarray(transformed)[0]
        indices = np.flatnonzero(dense_row)
        values = dense_row[indices]

    active_rows = []
    for index, value in zip(indices, values):
        if int(index) >= vectorizer_count:
            continue
        coefficient = float(coefficients[int(index)])
        active_value = float(value)
        impact = coefficient * active_value
        if impact == 0:
            continue
        active_rows.append(
            {
                "feature": manuscript_feature_name(feature_names[int(index)]),
                "value": active_value,
                "coefficient": coefficient,
                "impact": impact,
                "direction": "Fake" if impact < 0 else "Real",
            }
        )

    fake_rows = sorted(
        (row for row in active_rows if row["impact"] < 0),
        key=lambda row: row["impact"],
    )[:limit_per_direction]
    real_rows = sorted(
        (row for row in active_rows if row["impact"] > 0),
        key=lambda row: row["impact"],
        reverse=True,
    )[:limit_per_direction]
    return fake_rows + real_rows


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def table13_markdown(fp_features: dict[str, float], fn_features: dict[str, float]) -> str:
    rows = []
    for feature in TABLE13_ORDER:
        rows.append(
            [
                DISPLAY_NAMES[feature],
                format_number(fp_features[feature]),
                format_number(fn_features[feature]),
            ]
        )
    return markdown_table(
        ["Predictor", "False Positive (new Article 1)", "False Negative (Article 3)"],
        rows,
    )


def table14_markdown(fp_rows: list[dict[str, object]], fn_rows: list[dict[str, object]]) -> str:
    rows = []
    for article_label, predictor_rows in (
        ("False Positive", fp_rows),
        ("False Negative", fn_rows),
    ):
        for row in predictor_rows:
            rows.append(
                [
                    article_label,
                    row["direction"],
                    row["feature"],
                    format_number(float(row["coefficient"])),
                    format_number(float(row["value"])),
                    format_number(float(row["impact"])),
                ]
            )
    return markdown_table(
        ["Article", "Direction", "Predictor", "Coefficient", "Value", "Impact"],
        rows,
    )


def choose_false_positive(results: list[dict[str, object]]) -> dict[str, object]:
    false_positives = [row for row in results if row["predicted"] == 1]
    if not false_positives:
        raise RuntimeError("No false positive found; compose another candidate batch.")

    preferred_ids = ["cand_01", "cand_06", "cand_02", "cand_08", "cand_11"]
    by_id = {row["candidate"].article_id: row for row in false_positives}
    for article_id in preferred_ids:
        if article_id in by_id:
            return by_id[article_id]
    return max(false_positives, key=lambda row: row["real_probability"])


def build_report(
    selected: dict[str, object],
    fp_features: dict[str, float],
    fn_features: dict[str, float],
    fp_vectors: list[dict[str, object]],
    fn_vectors: list[dict[str, object]],
    fn_prediction: dict[str, object],
    false_positive_count: int,
) -> str:
    candidate = selected["candidate"]
    real_terms = ", ".join(
        row["feature"] for row in fp_vectors if row["direction"] == "Real"
    )
    fake_terms = ", ".join(
        row["feature"] for row in fp_vectors if row["direction"] == "Fake"
    )
    return "\n\n".join(
        [
            "# New False Positive Article",
            (
                "This article is freshly composed for deployment-model probing and "
                "is intentionally fabricated. It should replace Article 1 only in "
                "the deployment-test appendix."
            ),
            "## Selected Filipino Text\n\n" + candidate.text,
            "## English Translation\n\n" + candidate.translation,
            (
                "## Classification Result\n\n"
                f"- Candidate: {candidate.article_id} ({candidate.title})\n"
                f"- Word count: {word_count(candidate.text)}\n"
                "- Gold label: Fake (0)\n"
                f"- Predicted label: {'Real (1)' if selected['predicted'] == 1 else 'Fake (0)'}\n"
                f"- Probability Fake: {selected['fake_probability']:.6f}\n"
                f"- Probability Real: {selected['real_probability']:.6f}\n"
                f"- Candidate batch false positives: {false_positive_count} of {len(CANDIDATES)}\n"
                "\n"
                "Existing Article 3 remains a false negative:\n"
                f"- Gold label: Real (1)\n"
                f"- Predicted label: {'Real (1)' if fn_prediction['predicted'] == 1 else 'Fake (0)'}\n"
                f"- Probability Fake: {fn_prediction['fake_probability']:.6f}\n"
                f"- Probability Real: {fn_prediction['real_probability']:.6f}"
            ),
            "## Table 13 Replacement - Linguistic Predictors\n\n"
            + table13_markdown(fp_features, fn_features),
            (
                "## Table 14 Replacement - Active Vectorizer Predictors\n\n"
                "Rows are selected by the largest signed active contribution "
                "`coefficient * feature_value` among non-zero TF-IDF/BOW features.\n\n"
                + table14_markdown(fp_vectors, fn_vectors)
            ),
            (
                "## Analytical Note\n\n"
                f"The selected fabricated article is classified as Real mainly because "
                f"it uses a cautious journalistic register, has relatively low values "
                f"on several negative-weight linguistic predictors such as "
                f"ave-phrase-count ({format_number(fp_features['ave_phrase_count'])}) "
                f"and consonant-cluster ({format_number(fp_features['consonant_cluster'])}), "
                f"and activates real-leaning vectorizer terms such as {real_terms}. "
                f"Its active fake-leaning terms, including {fake_terms}, do not offset "
                f"the positive evidence enough, leaving the model at a Real probability "
                f"of {selected['real_probability']:.6f}. This is a useful blind spot for "
                f"the manuscript because the article is not an obvious keyword-stuffed "
                f"attack; it reads like a cautious local news brief built around "
                f"attribution, agency references, and a still-unconfirmed advisory."
            ),
        ]
    ) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("# Compose New False Positive Article")
    print(f"Repo: {REPO_DIR}")
    print(f"Model: {MODEL_PATH}")
    print(f"Candidate count: {len(CANDIDATES)}")

    pipeline = load_pipeline()
    print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
    print(
        "FeatureUnion transformers: "
        f"{[name for name, _ in pipeline.named_steps['features'].transformer_list]}"
    )

    dataset_texts = read_dataset_articles()
    print(f"Dataset exact-match texts loaded: {len(dataset_texts)}")

    results = []
    for candidate in CANDIDATES:
        result = predict_article(pipeline, candidate.text)
        result["candidate"] = candidate
        result["word_count"] = word_count(candidate.text)
        result["exact_dataset_match"] = normalize(candidate.text) in dataset_texts
        results.append(result)
        print(
            "Candidate "
            f"{candidate.article_id} | words={result['word_count']} | "
            f"predicted={result['predicted']} "
            f"({'Real' if result['predicted'] == 1 else 'Fake'}) | "
            f"proba_fake={result['fake_probability']:.6f} | "
            f"proba_real={result['real_probability']:.6f} | "
            f"dataset_exact_match={result['exact_dataset_match']} | "
            f"title={candidate.title}"
        )
        if not 100 <= result["word_count"] <= 200:
            print(f"WARNING: {candidate.article_id} is outside the 100-200 word target.")

    if any(row["exact_dataset_match"] for row in results):
        raise RuntimeError("At least one candidate exactly matches a dataset article.")

    false_positives = [row for row in results if row["predicted"] == 1]
    print(f"\nFalse positives found: {len(false_positives)}")
    for row in false_positives:
        candidate = row["candidate"]
        print(
            f"FP candidate {candidate.article_id} | "
            f"proba_real={row['real_probability']:.6f} | title={candidate.title}"
        )

    selected = choose_false_positive(results)
    selected_candidate = selected["candidate"]
    print(
        f"\nSelected FP: {selected_candidate.article_id} | "
        f"proba_real={selected['real_probability']:.6f} | "
        f"title={selected_candidate.title}"
    )

    fn_prediction = predict_article(pipeline, FN_TEXT)
    print(
        "Existing FN Article 3 | "
        f"predicted={fn_prediction['predicted']} "
        f"({'Real' if fn_prediction['predicted'] == 1 else 'Fake'}) | "
        f"proba_fake={fn_prediction['fake_probability']:.6f} | "
        f"proba_real={fn_prediction['real_probability']:.6f}"
    )

    fp_features = extract_linguistic_features(pipeline, selected_candidate.text)
    fn_features = extract_linguistic_features(pipeline, FN_TEXT)
    fp_vectors = active_vectorizer_predictors(pipeline, selected_candidate.text)
    fn_vectors = active_vectorizer_predictors(pipeline, FN_TEXT)

    print("\nTable 13 replacement:")
    print(table13_markdown(fp_features, fn_features))
    print("\nTable 14 replacement:")
    print(table14_markdown(fp_vectors, fn_vectors))

    report = build_report(
        selected=selected,
        fp_features=fp_features,
        fn_features=fn_features,
        fp_vectors=fp_vectors,
        fn_vectors=fn_vectors,
        fn_prediction=fn_prediction,
        false_positive_count=len(false_positives),
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nSaved report: {REPORT_PATH}")
    print(f"Saved log: {LOG_PATH}")


if __name__ == "__main__":
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            main()
