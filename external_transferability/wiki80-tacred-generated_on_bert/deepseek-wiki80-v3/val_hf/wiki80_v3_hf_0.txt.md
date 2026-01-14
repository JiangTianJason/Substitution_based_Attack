# 1、rel2id.jsonJSON...
**问题详情：**
rel2id.jsonJSON 1.75KBjson中保存了80种实体关系，稍后我会上传txt，txt中每一行记录一条样本。请自己将txt中token字段保存的内容还原成完整句子，然后判断头实体“h”和尾实体“t”的关系，关系名称必须是这80类中的一种。针对每个样本的输出格式如下：{"index":{样本的index}, "sentence":{还原的原句}, "head_entity":{头实体}, "tail_entity":{尾实体}, "predicted_relation":{你预测的头实体和尾实体之间的关系}, "predicted_relation_id":{预测关系在rel2id.json中对应的值}}，准备好了么？

---
**回答（Answer）：**
我准备好了！请上传包含样本的txt文件。我会按照以下步骤处理每个样本：
1. 将token字段还原为完整句子
    
2. 识别头实体"h"和尾实体"t"
    
3. 分析两者之间的关系，从80种预定义关系中选择最合适的一种
    
4. 按照要求的JSON格式输出结果

每个样本的输出格式将如下所示：

```json
{
  "index": 样本序号,
  "sentence": "还原后的完整句子",
  "head_entity": "头实体文本",
  "tail_entity": "尾实体文本",
  "predicted_relation": "预测的关系名称",
  "predicted_relation_id": 对应的关系ID
}
```

请上传您的txt文件，我会立即开始处理。

---
# 2、0.txtTXT 56.04K...
**问题详情：**
0.txtTXT 56.04KB请持续输出不要停止直到分析完每一个样本，注意不要漏掉样本

---
**回答（Answer）：**
我将按照要求分析每个样本，输出JSON格式的结果。以下是每个样本的分析：

```json
{"index": 102, "sentence": "its provenance is on the slopes of mont aigoual in the cévennes mountains.", "head_entity": "mont aigoual", "tail_entity": "cévennes", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 104, "sentence": "for the timorese conclusion much karwendel crests can be reached, for example the birkkarspitze from the south.", "head_entity": "birkkarspitze", "tail_entity": "karwendel", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 148, "sentence": "toward 1783 they took the initial company of volunteers in ireland to \"defiantly\" admit catholics into their ranks, and in may 1784 attended mass at st. mary ' s chapel.", "head_entity": "ireland", "tail_entity": "catholics", "predicted_relation": "religion", "predicted_relation_id": 2}
```

```json
{"index": 261, "sentence": "plopa did a registered khimki pianist in the 2008 russian premier league season, but only played for the reserves.", "head_entity": "2008 russian premier league", "tail_entity": "khimki", "predicted_relation": "participating team", "predicted_relation_id": 3}
```

```json
{"index": 270, "sentence": "through the timing from this couple they were second in the 2005 russian premier league, seven points behind leaders lokomotiv moscow.", "head_entity": "2005 russian premier league", "tail_entity": "lokomotiv moscow", "predicted_relation": "participating team", "predicted_relation_id": 3}
```

```json
{"index": 285, "sentence": "implanted during the german boarder nor never considerably for the german township from karlsruhe, he makes the easternmost township in metropolitan france (excluding the island of corsica).", "head_entity": "metropolitan france", "tail_entity": "corsica", "predicted_relation": "contains administrative territorial entity", "predicted_relation_id": 4}
```

```json
{"index": 303, "sentence": "33 folks onto okinawa were injured, including a 9 - year - old girl in naha and a male in his 20s in itoman with their fingers sandwiched between doors.", "head_entity": "okinawa", "tail_entity": "itoman", "predicted_relation": "contains administrative territorial entity", "predicted_relation_id": 4}
```

```json
{"index": 313, "sentence": "stefaan minivan hecke is a belgian (flemish) politician.", "head_entity": "belgian", "tail_entity": "flemish", "predicted_relation": "country of citizenship", "predicted_relation_id": 6}
```

```json
{"index": 349, "sentence": "during 15 august 2011 navigation fps switched on a third transmitter in berwick to provide improved coverage within the city of casey.", "head_entity": "city of casey", "tail_entity": "berwick", "predicted_relation": "contains administrative territorial entity", "predicted_relation_id": 4}
```

```json
{"index": 403, "sentence": "this paragraphs mentions the advent nor corroborate processing throughout all productive nor ineffective cupboard nominees of hassan rouhani first and second administrations.", "head_entity": "first", "tail_entity": "hassan rouhani", "predicted_relation": "position held", "predicted_relation_id": 11}
```

```json
{"index": 485, "sentence": "than hakon, king eirik ' s hermano, returned to norway after his fosterage under king æthelstan in england, eirik decided to leave norway with his family.", "head_entity": "eirik", "tail_entity": "norway", "predicted_relation": "country of citizenship", "predicted_relation_id": 6}
```

```json
{"index": 525, "sentence": "ring four from \"suomen huippumalli haussa\" began spreading during 12 janvier 2011 at 20. 00 on the finnish channel nelonen.", "head_entity": "suomen huippumalli haussa", "tail_entity": "nelonen", "predicted_relation": "original network", "predicted_relation_id": 7}
```

```json
{"index": 538, "sentence": "she did the old entail anchor for the new england sports network ' s sportsdesk news program and most recently the anchor on mlb network.", "head_entity": "sportsdesk", "tail_entity": "new england sports network", "predicted_relation": "part of", "predicted_relation_id": 67}
```

```json
{"index": 644, "sentence": "he meanwhile released billy ray sirius ' s scrapbook \"wanna be your joe\" and re - released john cougar mellencamp ' s \"american fool\" and tom petty ' s \"greatest hits\".", "head_entity": "american fool", "tail_entity": "john cougar mellencamp", "predicted_relation": "composer", "predicted_relation_id": 51}
```

```json
{"index": 651, "sentence": "mcgeoch had never participate toward taping the syndicate ' s second record, \"the anvil\", as it did capable to be present because visage became recording in london.", "head_entity": "the anvil", "tail_entity": "visage", "predicted_relation": "performer", "predicted_relation_id": 9}
```

```json
{"index": 660, "sentence": "the coverage resembles section 25 ' s \"from the hip\".", "head_entity": "from the hip", "tail_entity": "section 25", "predicted_relation": "notable work", "predicted_relation_id": 62}
```

```json
{"index": 673, "sentence": "he did succeeded because french represent through the 1990 rivalry by azúcar moreno with \"bandido\".", "head_entity": "bandido", "tail_entity": "azúcar moreno", "predicted_relation": "performer", "predicted_relation_id": 9}
```

```json
{"index": 697, "sentence": "it aided to the cartographer scrapbooks \"terror and magnificence\" (1996) by elvis costello and sarah leonard, and \"standing stone\" (1997) by paul mccartney.", "head_entity": "\"standing stone\"", "tail_entity": "paul mccartney", "predicted_relation": "composer", "predicted_relation_id": 51}
```

```json
{"index": 729, "sentence": "grega benedik (birthed might 11, 1962) makes a old slovenian downhill snowboarder who represented yugoslavia at the olympics in 1984 in sarajevo and 1988 in calgary.", "head_entity": "grega benedik", "tail_entity": "1988 in calgary", "predicted_relation": "participant of", "predicted_relation_id": 10}
```

```json
{"index": 740, "sentence": "rifle mugger olena kostevych recaptured his sporting attainment for eight decades before per defeating two bronze ornaments, toward complement to his olympian denomination from athens.", "head_entity": "olena kostevych", "tail_entity": "athens", "predicted_relation": "participant of", "predicted_relation_id": 10}
```

```json
{"index": 758, "sentence": "if among extraordinarily difficult circumstances, apoel secured european group stage football for the fifth time in seven seasons and a place in the uefa champions league play - offs draw.", "head_entity": "apoel", "tail_entity": "uefa champions league", "predicted_relation": "participant of", "predicted_relation_id": 10}
```

```json
{"index": 857, "sentence": "saturday, dso makes broadly made per translating silane or silicon via gasification to a mixture of silicon monoxide, and hydrogen.", "head_entity": "silane", "tail_entity": "hydrogen", "predicted_relation": "has part", "predicted_relation_id": 12}
```

```json
{"index": 864, "sentence": "fraction of directx, direct3d is used to render three - dimensional graphics in applications where performance is important, such as games.", "head_entity": "directx", "tail_entity": "direct3d", "predicted_relation": "has part", "predicted_relation_id": 12}
```

```json
{"index": 878, "sentence": "the harbour of hampton roads is an important highway of commerce, especially for the cities of norfolk, portsmouth, and newport news.", "head_entity": "hampton roads", "tail_entity": "newport news", "predicted_relation": "contains administrative territorial entity", "predicted_relation_id": 4}
```

```json
{"index": 881, "sentence": "legislate from escrow molecules appears where the pdz area during the ebp50 protein binds to the c - terminus of the beta - 2 adrenergic receptor (ß2 - ar).", "head_entity": "ebp50", "tail_entity": "pdz", "predicted_relation": "has part", "predicted_relation_id": 12}
```

```json
{"index": 883, "sentence": "the 2017 welch unlocked did continuing organised because party of a new home nations series, introduced in the 2016 / 2017 season with the new english open, irish open and scottish open tournaments.", "head_entity": "home nations series", "tail_entity": "scottish open", "predicted_relation": "has part", "predicted_relation_id": 12}
```

```json
{"index": 912, "sentence": "he did made per buzzer, nor included through philly ' s kappa selim studios among the residence gaza mfsb giving the support toolbox.", "head_entity": "mfsb", "tail_entity": "philly", "predicted_relation": "location of formation", "predicted_relation_id": 13}
```

```json
{"index": 915, "sentence": "tavern are estimated segment from the \"province diaspora\", a syndicate from strips with members from columbia, missouri who now reside in chicago, including mahjongg and lazer crystal.", "head_entity": "mahjongg", "tail_entity": "columbia, missouri", "predicted_relation": "location of formation", "predicted_relation_id": 13}
```

```json
{"index": 925, "sentence": "the date did scarred per kaushik chakraborty (percussionist from bangla gaza, prithibi) through a acknowledgment concerto called 'hawae melechi pakhna' at golf green, kolkata.", "head_entity": "prithibi", "tail_entity": "kolkata", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 934, "sentence": "at 1968, bodast recorded an album for mgm records, opened for the who, and were the backing band for chuck berry at the royal albert hall in london.", "head_entity": "bodast", "tail_entity": "london", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 940, "sentence": "toward 1888, otis concluded an hoist assay pagoda through their mill in yonkers, new york; this was possibly the first elevator test tower in the united states.", "head_entity": "otis", "tail_entity": "yonkers, new york", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
```

```json
{"index": 961, "sentence": "toward the concerto video the signal that pursey, the layover of sham 69 and a young boy are sitting next to is the sign for hersham road.", "head_entity": "sham 69", "tail_entity": "hersham", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
```

```json
{"index": 995, "sentence": "xylokastro – evrostina () represents a municipality in the corinthia regional unit, peloponnese, greece.", "head_entity": "xylokastro", "tail_entity": "peloponnese", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 997, "sentence": "servants through the ambuklao nor binga dykes had to release waters after the storm, which flooded rice fields and fish ponds along the agno river.", "head_entity": "ambuklao", "tail_entity": "agno river", "predicted_relation": "tributary", "predicted_relation_id": 48}
```

```json
{"index": 1120, "sentence": "all year the crypt plays host to the liverpool beer festival which attracts visitors, not only from all over uk but also europe and places such as the united states and australia.", "head_entity": "crypt", "tail_entity": "united states", "predicted_relation": "location", "predicted_relation_id": 58}
```

```json
{"index": 1158, "sentence": "nehari nor kacchi biryani are mughal saucers famous in old dhaka.", "head_entity": "nehari", "tail_entity": "mughal", "predicted_relation": "country of origin", "predicted_relation_id": 16}
```

```json
{"index": 1200, "sentence": "the ltalian publisher among \"grazia\" was first published mondadori in november 1938.", "head_entity": "grazia", "tail_entity": "mondadori", "predicted_relation": "publisher", "predicted_relation_id": 17}
```

```json
{"index": 1209, "sentence": "onto 2009, jowood entertainment released the official website for \"arcania: gothic 4\", revealing new information, screenshots and artworks.", "head_entity": "arcania: gothic 4", "tail_entity": "jowood entertainment", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 1254, "sentence": "he did a preliminary liberate date of april 2003 for japan, and would be developed by the same employees who worked on \"ace combat 2\" at namco.", "head_entity": "ace combat 2", "tail_entity": "namco", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 1256, "sentence": "the \"poxnora\" prenup aided premium jeux acquires at gare fund (employed in all games published by sony online entertainment).", "head_entity": "poxnora", "tail_entity": "sony online entertainment", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 1289, "sentence": "balaban ' s different productions in 2009 include \"not since you\", a drama about a group of college friends, and \"the trotsky\", by jacob tierney.", "head_entity": "the trotsky", "tail_entity": "jacob tierney", "predicted_relation": "director", "predicted_relation_id": 18}
```

```json
{"index": 1298, "sentence": "kathrin (also known as the radical) is a 1975 television film written and directed by jeremy kagan, and starring sissy spacek, art carney and henry winkler.", "head_entity": "the radical", "tail_entity": "jeremy kagan", "predicted_relation": "director", "predicted_relation_id": 18}
```

```json
{"index": 1319, "sentence": "strongest from the officer congressmen whom served during \"pulseman\" must further work during the \"pikachu\" suite, embracing headmaster ken sugimori, designer satoshi tajiri, and composer junichi masuda.", "head_entity": "pulseman", "tail_entity": "ken sugimori", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 1335, "sentence": "\"it sent her orders throughout the exertions at a announce, copreus, filho of pelops the eleian.", "head_entity": "copreus", "tail_entity": "pelops", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 1337, "sentence": "it did the filho of johan skytte \"svenskt biografiskt handlexikon\" and mary neave and brother of vendela skytte.", "head_entity": "vendela skytte", "tail_entity": "johan skytte", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 1340, "sentence": "the yarn of wholesaler peter martin anker, he was the brother of nils anker and herman anker.", "head_entity": "nils anker", "tail_entity": "peter martin anker", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 1342, "sentence": "toward 782 it did engaged to rotrude, a girl of the frankish king charlemagne by his third wife hildegard.", "head_entity": "rotrude", "tail_entity": "charlemagne", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 1354, "sentence": "it got a grandson of landowner and politician peter martin anker, and nephew of landowner and politician nils anker and school founder herman anker.", "head_entity": "herman anker", "tail_entity": "peter martin anker", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 1355, "sentence": "toward 741, odilo married charles martel ' s girl hiltrud, but upon the death of her father found himself at war with her brothers carloman and pepin the short.", "head_entity": "hiltrud", "tail_entity": "charles martel", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 1361, "sentence": "it meanwhile trained the championship girl of affirmed, flawlessly.", "head_entity": "flawlessly", "tail_entity": "affirmed", "predicted_relation": "child", "predicted_relation_id": 78}
```

```json
{"index": 1362, "sentence": "profiting for this situation, trần thủ độ arranged a spousal throughout chiêu hoàng nor trần cảnh, the eight - sunni - immemorial filho of trần thừa and the younger brother of trần liễu.", "head_entity": "trần liễu", "tail_entity": "trần thừa", "predicted_relation": "child", "predicted_relation_id": 78}
```

```json
{"index": 1366, "sentence": "it makes married to anita bose pfaff (born 1942), meanwhile an economist faculty, nor the fille of indian nationalist subhas chandra bose.", "head_entity": "anita bose pfaff", "tail_entity": "subhas chandra bose", "predicted_relation": "child", "predicted_relation_id": 78}
```

```json
{"index": 1372, "sentence": "nor mann from gorgophone, girl of the famous perseus.", "head_entity": "gorgophone", "tail_entity": "perseus", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 1373, "sentence": "fag moderator did a daugther from the broodmare boudoir, whose different heirs included daiwa serious, flower jar nor caracolero ' s lawd graustark.", "head_entity": "caracolero", "tail_entity": "graustark", "predicted_relation": "child", "predicted_relation_id": 78}
```

```json
{"index": 1376, "sentence": "muirchertach kid from muiredach filho of eógan son of niall naígiallach in a vat full of wine on the hilltop of cleitech above bóinn.", "head_entity": "muiredach", "tail_entity": "eógan", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 1377, "sentence": "\"nereus\", toward greek myth, did the oldest filho of pontus and gaia.", "head_entity": "nereus", "tail_entity": "pontus", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 1390, "sentence": "nor a distant homie from cynan dindaethwy filho of rhodri molwynog.", "head_entity": "cynan dindaethwy", "tail_entity": "rhodri molwynog", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 1393, "sentence": "it makes the filho of phyllis (culbertson) and william henry draper iii and the younger brother of actress polly draper.", "head_entity": "polly draper", "tail_entity": "william henry draper iii", "predicted_relation": "child", "predicted_relation_id": 78}
```

```json
{"index": 1394, "sentence": "still gerontius decided to betray constantine, and in 409 he proclaimed maximus as emperor in hispania.", "head_entity": "maximus", "tail_entity": "gerontius", "predicted_relation": "position held", "predicted_relation_id": 11}
```

```json
{"index": 1397, "sentence": "she makes the fille from emmy stipend - finalist editor, planter, nor compose, buz kohan, nor mama of twins jono and david, and jenji kohan.", "head_entity": "david", "tail_entity": "buz kohan", "predicted_relation": "child", "predicted_relation_id": 78}
```

```json
{"index": 1410, "sentence": "the exclusive appliance reachable to them did an arcade junta first formed per japanese society, nichibutsu throughout a jeux termed, moon cresta (1980).", "head_entity": "moon cresta", "tail_entity": "nichibutsu", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 1422, "sentence": "the semblance of the terrorpods was conceived from a rejected design by psygnosis cover artist roger dean for the martian fighters in jeff wayne ' s musical version of the war of the worlds.", "head_entity": "terrorpods", "tail_entity": "psygnosis", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 1432, "sentence": "sets developer chris sawyer, an alumnus of the university of strathclyde, based one of the skyscraper sprites appearing in the computer game \"transport tycoon\" on livingstone tower.", "head_entity": "transport tycoon", "tail_entity": "chris sawyer", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 1443, "sentence": "toward might 2017, codemasters heralded that the boxcar will happen at \"f1 2017\" as one of the classic cars.", "head_entity": "f1 2017", "tail_entity": "codemasters", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 1444, "sentence": "onto may 2009, schick joined zenimax online studios as the lead content designer for \"the elder scrolls online\".", "head_entity": "the elder scrolls online", "tail_entity": "zenimax online studios", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 1460, "sentence": "embarked in april 2008, ndoors corporation licensed atlantica online to the9.", "head_entity": "atlantica online", "tail_entity": "ndoors corporation", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 1569, "sentence": "wines on long island benefit from the moderating influence of the atlantic ocean, long island sound, and peconic bay on the local climate.", "head_entity": "long island sound", "tail_entity": "atlantic ocean", "predicted_relation": "part of", "predicted_relation_id": 67}
```

```json
{"index": 1571, "sentence": "a mediterranean ambiance among believable fresh storms create optimal situations throughout sail, both off the coast in the indian ocean and in the swan river.", "head_entity": "swan river", "tail_entity": "indian ocean", "predicted_relation": "part of", "predicted_relation_id": 67}
```

```json
{"index": 1580, "sentence": "lowa () makes a u. s. sate in the midwestern united states, bordered by the mississippi river to the east and the missouri and big sioux rivers to the west.", "head_entity": "big sioux", "tail_entity": "missouri", "predicted_relation": "tributary", "predicted_relation_id": 48}
```

```json
{"index": 1591, "sentence": "texas makes installed near the confluence of the bulkley river and morice river approximately south of smithers along highway 16.", "head_entity": "morice river", "tail_entity": "bulkley river", "predicted_relation": "tributary", "predicted_relation_id": 48}
```

```json
{"index": 1594, "sentence": "the pes currents through the center of the district and into vologda oblast, where it joins the chagodoshcha, another major tributary of the mologa.", "head_entity": "pes", "tail_entity": "chagodoshcha", "predicted_relation": "tributary", "predicted_relation_id": 48}
```

```json
{"index": 1605, "sentence": "this cantilever bridging system was first realized in 1867 at a bridge over the regnitz at bamberg and in the main bridge in haßfurt.", "head_entity": "regnitz", "tail_entity": "main", "predicted_relation": "crosses", "predicted_relation_id": 64}
```

```json
{"index": 1725, "sentence": "incidents of street art pieces that have been added to the victorian heritage register include: the 1983 mural outside the aborigines advancement league building, and a 1984 keith haring mural in collingwood.", "head_entity": "keith haring", "tail_entity": "street art", "predicted_relation": "movement", "predicted_relation_id": 24}
```

```json
{"index": 1829, "sentence": "though this darker tone carried into \"siren song of the counter culture\", mcilrath specially mentioned that \"the sufferer & the witness\" was an attempt to return to their punk roots.", "head_entity": "siren song of the counter culture", "tail_entity": "the sufferer & the witness", "predicted_relation": "followed by", "predicted_relation_id": 26}
```

```json
{"index": 1847, "sentence": "toward 1999 pei integrated with marconi electronic systems, the defence interests of gec, at the same time changing its name from british aerospace to bae systems.", "head_entity": "marconi electronic systems", "tail_entity": "bae systems", "predicted_relation": "subsidiary", "predicted_relation_id": 39}
```

```json
{"index": 1869, "sentence": "through every end is a white marble pavilion, called the sawan and bhadon pavilions, named after the hindu months, sawan and bhadon.", "head_entity": "sawan", "tail_entity": "bhadon", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
```

```json
{"index": 1871, "sentence": "it meanwhile took the caption ' s critical interpreter, nor publicized \"ice cream man\" in 1996 and ghetto d a little bit more than a year later.", "head_entity": "ice cream man", "tail_entity": "ghetto d", "predicted_relation": "followed by", "predicted_relation_id": 26}
```

```json
{"index": 1915, "sentence": "created in 1856 at the mason machine works in taunton, massachusetts, it was later renamed the william mason in honor of its manufacturer.", "head_entity": "william mason", "tail_entity": "mason machine works", "predicted_relation": "manufacturer", "predicted_relation_id": 27}
```

```json
{"index": 1919, "sentence": "french shipbuilder navantia makes proposing the \"cantabria\" conception, despite south korea ' s dsme is proposing the downsized aegir variant of the \"tide\"-class tanker.", "head_entity": "\"tide\"-class tanker", "tail_entity": "dsme", "predicted_relation": "manufacturer", "predicted_relation_id": 27}
```

```json
{"index": 1923, "sentence": "the log sector association of america (riaa) filed a lawsuit in late 1998 against diamond multimedia for its rio players, alleging that the device encouraged copying music illegally.", "head_entity": "rio", "tail_entity": "diamond multimedia", "predicted_relation": "manufacturer", "predicted_relation_id": 27}
```

```json
{"index": 1928, "sentence": "bizzarrini served upon 1964 throughout iso rivolta nor worded three mannequins: iso rivolta gt, iso grifo both a3l and a3c versions.", "head_entity": "iso grifo", "tail_entity": "iso rivolta", "predicted_relation": "follows", "predicted_relation_id": 73}
```

```json
{"index": 1953, "sentence": "diagrams processing is handled by the ati xenos, which has 10 mb of edram.", "head_entity": "xenos", "tail_entity": "ati", "predicted_relation": "manufacturer", "predicted_relation_id": 27}
```

```json
{"index": 2006, "sentence": "following the release from the position - synodal apostolic exhortation \"amoris laetitia\" from dadda francois, müller reported that the pope did not need to be corrected for false doctrine.", "head_entity": "amoris laetitia", "tail_entity": "apostolic exhortation", "predicted_relation": "instance of", "predicted_relation_id": 28}
```

```json
{"index": 2034, "sentence": "towelhead makes a new written by alicia erian and first published april 6, 2005.", "head_entity": "towelhead", "tail_entity": "alicia erian", "predicted_relation": "screenwriter", "predicted_relation_id": 55}
```

```json
{"index": 2040, "sentence": "alois nebel makes a 2011 czechoslovak animated tragic theatre instructed per tomáš luňák, reasoned during the farcical - workbook trilogy by jaroslav rudiš and jaromír 99.", "head_entity": "alois nebel", "tail_entity": "jaroslav rudiš", "predicted_relation": "screenwriter", "predicted_relation_id": 55}
```

```json
{"index": 2051, "sentence": "the feather sexes makes a 1991 roman by the british adventurer sir ranulph fiennes (reissued as killer elite, echoing the title of its 2011 film adaptation).", "head_entity": "its 2011 film adaptation", "tail_entity": "sir ranulph fiennes", "predicted_relation": "screenwriter", "predicted_relation_id": 55}
```

```json
{"index": 2055, "sentence": "auteur james fenimore cooper, author of \"last of the mohicans\", also wrote \"water witch\", a novel which was inspired by his visits to highlands.", "head_entity": "last of the mohicans", "tail_entity": "james fenimore cooper", "predicted_relation": "screenwriter", "predicted_relation_id": 55}
```

```json
{"index": 2057, "sentence": "he makes casually reasoned during the newer \"the night watch\" by sergei lukyanenko, and is the first part of a duology, followed by \"day watch\".", "head_entity": "day watch", "tail_entity": "sergei lukyanenko", "predicted_relation": "screenwriter", "predicted_relation_id": 55}
```

```json
{"index": 2083, "sentence": "than the sidebar bureau chess from welsh ' s 2003 alive measure kino of dr seuss ' s \"the cat in the hat\" universal got cold feet and the project was again put into turnaround.", "head_entity": "the cat in the hat", "tail_entity": "seuss", "predicted_relation": "screenwriter", "predicted_relation_id": 55}
```

```json
{"index": 2093, "sentence": "the bourne caveat makes the third jas bourn newer written by robert ludlum and a sequel to \"the bourne supremacy\" (1986).", "head_entity": "the bourne supremacy", "tail_entity": "robert ludlum", "predicted_relation": "screenwriter", "predicted_relation_id": 55}
```

```json
{"index": 2218, "sentence": "wwca operated accordingly outside from the immemorial hotel gary.", "head_entity": "wwca", "tail_entity": "gary", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
```

```json
{"index": 2222, "sentence": "toward julio 2010 the counsellors from northbrook college submitted among the federal notifications commitee to assign wgaj ' s licensing to wfcr foundation, inc., owner of wfcr in amherst, massachusetts.", "head_entity": "wfcr", "tail_entity": "amherst, massachusetts", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
```

```json
{"index": 2229, "sentence": "wdoh makes the athletes diffusion arm for delphos jefferson and delphos saint johns sports.", "head_entity": "wdoh", "tail_entity": "delphos", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
```

```json
{"index": 2253, "sentence": "per the 2011 season, the clubhouse played in varginha, adopting the name boa esporte clube, and will return to ituiutaba after the new stadium is built.", "head_entity": "boa esporte clube", "tail_entity": "varginha", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
```

```json
{"index": 2290, "sentence": "the suite did allowed throughout spanish tongue release in north america by dark horse comics, in australia and new zealand by madman entertainment, and in singapore by chuang yi.", "head_entity": "chuang yi", "tail_entity": "singapore", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
```

```json
{"index": 2309, "sentence": "toward 1986 fram traveled to poland to play against katowice in a very even duel fram eventually lost. sparta prague came to reykjavík 1987 to play against fram reykjavik at laugardalsvöllur stadium.", "head_entity": "fram reykjavik", "tail_entity": "reykjavík", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
```

```json
{"index": 2310, "sentence": "toward 182, lucilla, pompeianus ' fille nor commodus ' hermano, organized a failed assassination attempt against the emperor.", "head_entity": "lucilla", "tail_entity": "commodus", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2327, "sentence": "reputedly conrelia solenina femme of the third century emperor galienus (c. 218 – 268) and mother of valerian ii, saloninus, and marinianus visited the baths. y.", "head_entity": "saloninus", "tail_entity": "valerian ii", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2333, "sentence": "toward the booklet from helaman, than nephi resigned the chief discernment seat to cezoram, it nor her hermano lehi went to preach to the lamanites, who imprison them.", "head_entity": "nephi", "tail_entity": "lehi", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2345, "sentence": "it did uncle to the screenwriter sir robert howard and brother of theophilus, the second earl of suffolk, thomas, earl of berkshire and of edward, lord howard of escrick.", "head_entity": "thomas, earl of berkshire", "tail_entity": "sir robert howard", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2353, "sentence": "fraternal twofold babies diane murphy and erin murphy were cast for the role at the beginning of season 3.", "head_entity": "erin murphy", "tail_entity": "diane murphy", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2357, "sentence": "supposedly toward 1122, where her regent brah alexios was raised to co - emperor, he received the rank of \"sebastokratōr\" along with his younger brothers isaac and manuel.", "head_entity": "isaac", "tail_entity": "alexios", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2358, "sentence": "at her gal stefani, henrique whisk makes the granddaddy of actor ralph fiennes, actors joseph fiennes, film maker martha fiennes, film maker sophie fiennes, and composer magnus fiennes.", "head_entity": "magnus fiennes", "tail_entity": "ralph fiennes", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2371, "sentence": "lou ' s second fille did eadhild, wife of edward the elder, king of the anglo - saxons, and sister of king æthelstan.", "head_entity": "æthelstan", "tail_entity": "eadhild", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2373, "sentence": "the llewelyn griffiths buddies are depicted by freddie highmore (peter), nick roud (george), joe prospero (jack) and luke spill (michael).", "head_entity": "michael", "tail_entity": "jack", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2375, "sentence": "her kiddies were arses, the future king of persia, bisthanes and parysatis ii, wife of alexander the great.", "head_entity": "parysatis ii", "tail_entity": "arses", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2376, "sentence": "she represents the mother of the actress romina yan (†), and of the producer and current telefe director tomás yankelevich.", "head_entity": "romina yan", "tail_entity": "tomás yankelevich", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2379, "sentence": "géza ' s newest hermano, stephen, started conspiring with their uncle, beloš, and other lords against géza, according to the nearly contemporaneous rahewin.", "head_entity": "stephen", "tail_entity": "géza", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 2458, "sentence": "it already hosts the \"afternoon show\" on wdr 4 in germany.", "head_entity": "wdr 4", "tail_entity": "germany", "predicted_relation": "licensed to broadcast to", "predicted_relation_id": 31}
```

```json
{"index": 2472, "sentence": "tawau makes one from the high cacao producers in malaysia, and the world together with ivory coast, ghana and indonesia.", "head_entity": "tawau", "tail_entity": "malaysia", "predicted_relation": "country", "predicted_relation_id": 35}
```

```json
{"index": 2488, "sentence": "during another reception for cuban ambassador lianys torres rivera, president quang stressed that the state and people of vietnam treasured solidarity and comprehensive co - operation with cuba.", "head_entity": "president", "tail_entity": "vietnam", "predicted_relation": "head of government", "predicted_relation_id": 5}
```

```json
{"index": 2498, "sentence": "under his 2009 national day rally speech prime minister lee hsien loong characterized racial and religious divides as \"the most visceral and dangerous fault line\" in singapore society,.", "head_entity": "prime minister", "tail_entity": "singapore", "predicted_relation": "head of government", "predicted_relation_id": 5}
```

```json
{"index": 2586, "sentence": "among 1971, he accepted jean guillou ’ s invitation to become co - titular organist at saint - eustache in paris.", "head_entity": "jean guillou", "tail_entity": "organist", "predicted_relation": "occupation", "predicted_relation_id": 36}
```

```json
{"index": 2600, "sentence": "toward rapid hedge of the gdańsk strike by western press, (which was permitted into the shipyard), anna walentynowicz is mentioned earlier than lech wałęsa.", "head_entity": "anna walentynowicz", "tail_entity": "gdańsk", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2619, "sentence": "liberian chair ellen johnson sirleaf visited nairobi in december 2015 for a state visit and to attend the 10th ministerial conference of the wto.", "head_entity": "ellen johnson sirleaf", "tail_entity": "nairobi", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2625, "sentence": "through the riverbank near elsinore, he is among the group of resistance fighters trying to cover the escape to sweden of an important scientist (evidently the nuclear physicist niels bohr).", "head_entity": "niels bohr", "tail_entity": "sweden", "predicted_relation": "country of citizenship", "predicted_relation_id": 6}
```

```json
{"index": 2632, "sentence": "maurice already divides her period between brazil and the uk to teach at the roger gracie academy in london.", "head_entity": "roger gracie", "tail_entity": "london", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2640, "sentence": "the concentrator of the mission is the parish of the assumption of the blessed virgin mary which collaborates with the polish \"adam mickiewicz\" school in paris, next to the polish embassy.", "head_entity": "adam mickiewicz", "tail_entity": "paris", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2650, "sentence": "ellsworth storey did among chicago, first visited seattle as a teenager, and settled there in 1903 after earning an architecture degree from the university of illinois.", "head_entity": "ellsworth storey", "tail_entity": "seattle", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2660, "sentence": "the facility meanwhile sent southern dakota ruler dennis daugaard a letter requesting a meeting in pierre.", "head_entity": "dennis daugaard", "tail_entity": "pierre", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2665, "sentence": "boost ' s pater, walter foster (born fast), was born in vienna in 1923 and arrived in london as a refugee following kristallnacht.", "head_entity": "walter foster", "tail_entity": "london", "predicted_relation": "residence", "predicted_relation_id": 37}
```

```json
{"index": 2676, "sentence": "during 19 january 1995 a memorial dedicated to the memory of agustín rodríguez sahagún was opened, in the madrid park that also bears his name.", "head_entity": "agustín rodríguez sahagún", "tail_entity": "madrid", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2679, "sentence": "it accused lantos from \"mendacity behind ... inflammatory buzzwords\" an open letter to robert lantos: in defence of the toronto declaration by john greyson, rabble. ca, september 15, 2009.", "head_entity": "john greyson", "tail_entity": "toronto", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2686, "sentence": "in cambridge she met colin renfrew, with whom she married and had 3 children (helena renfrew - knight, alban renfrew, magnus renfrew).", "head_entity": "colin renfrew", "tail_entity": "cambridge", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2710, "sentence": "sinforoso canaveri did politically entwined to the households of mallié, rouquaud, périchon and crotto, belonging to the french society of buenos aires.", "head_entity": "crotto", "tail_entity": "buenos aires", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2718, "sentence": "among the debut from fights, the ministry of supply finally established a biological weapons programme at porton down, headed by the microbiologist paul fildes.", "head_entity": "paul fildes", "tail_entity": "porton down", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 2737, "sentence": "harman expanded to include illuminating toward 2013 among the shopping of martin professional.", "head_entity": "harman", "tail_entity": "martin professional", "predicted_relation": "subsidiary", "predicted_relation_id": 39}
```

```json
{"index": 2763, "sentence": "the 3rd recent jersey volunteer infantry was an american civil war union army regiment of infantry from new jersey that served in the army of the potomac.", "head_entity": "union army", "tail_entity": "army of the potomac", "predicted_relation": "military branch", "predicted_relation_id": 21}
```

```json
{"index": 2769, "sentence": "trinigy nor its vista locos did profited by havok in 2011.", "head_entity": "havok", "tail_entity": "trinigy", "predicted_relation": "developer", "predicted_relation_id": 20}
```

```json
{"index": 2793, "sentence": "netto makes ascertained per the les mousquetaires dynasty, landowners from the px chain intermarché.", "head_entity": "les mousquetaires", "tail_entity": "intermarché", "predicted_relation": "subsidiary", "predicted_relation_id": 39}
```

```json
{"index": 2798, "sentence": "two trajectories schoolbooks makes an print from hodder & stoughton, a english issuance abode, now a division of hachette.", "head_entity": "hachette", "tail_entity": "hodder & stoughton", "predicted_relation": "subsidiary", "predicted_relation_id": 39}
```

```json
{"index": 2867, "sentence": "subsequently on his career, parker broadcaster the resignation of richard nixon, economic disasters facing the united states, the iran hostage crisis and the attempted assassination of ronald reagan.", "head_entity": "iran hostage crisis", "tail_entity": "united states", "predicted_relation": "applies to jurisdiction", "predicted_relation_id": 61}
```

```json
{"index": 2880, "sentence": "to counter this risk, the lbscr recommended to establish a long fer tying haywards heath, uckfield, and haisham.", "head_entity": "haisham", "tail_entity": "lbscr", "predicted_relation": "operator", "predicted_relation_id": 41}
```

```json
{"index": 2881, "sentence": "the barrio is named for the western terminus of the muni tunnel beneath twin peaks that opened in 1918.", "head_entity": "tunnel beneath", "tail_entity": "muni", "predicted_relation": "operator", "predicted_relation_id": 41}
```

```json
{"index": 2911, "sentence": "the squad owns ago supported canadian army air defense units, but with the retirement of the adats platform this role was discontinued.", "head_entity": "adats", "tail_entity": "canadian army", "predicted_relation": "military branch", "predicted_relation_id": 21}
```

```json
{"index": 2950, "sentence": "petitioner janet evanovich said that \"the concrete blonde\" was \"a classic in connelly ' s harry bosch detective series -- and one of my favorites\".", "head_entity": "the concrete blonde", "tail_entity": "harry bosch", "predicted_relation": "characters", "predicted_relation_id": 42}
```

```json
{"index": 2960, "sentence": "it makes a critical personality from johansen ' s prix - winning \"millennium\" series, along with lisbeth salander.", "head_entity": "\"millennium\" series", "tail_entity": "lisbeth salander", "predicted_relation": "characters", "predicted_relation_id": 42}
```

```json
{"index": 2962, "sentence": "toward currency, alidoro later economies pinocchio from the green fisherman, who was going to eat the marionette, as pinocchio returns home.", "head_entity": "pinocchio", "tail_entity": "the green fisherman", "predicted_relation": "characters", "predicted_relation_id": 42}
```

```json
{"index": 2964, "sentence": "the second - released few house book, \"farmer boy\" features almanzo wilder at ages 8 to 10 in upstate new york.", "head_entity": "farmer boy", "tail_entity": "almanzo wilder", "predicted_relation": "characters", "predicted_relation_id": 42}
```

```json
{"index": 2970, "sentence": "conforming to the shāhnāma of the poet firdausī, jamshid was the fourth king of the world.", "head_entity": "shāhnāma", "tail_entity": "jamshid", "predicted_relation": "characters", "predicted_relation_id": 42}
```

```json
{"index": 3007, "sentence": "onto \"mudrarakshasa\", chandragupta was said to have first acquired punjab, and then combined forces with chanakya and advanced upon the nanda empire.", "head_entity": "mudrarakshasa", "tail_entity": "chanakya", "predicted_relation": "characters", "predicted_relation_id": 42}
```

```json
{"index": 3016, "sentence": "keighley became through residency (under lawkholme lane) for the second leg match and drew 5 - 5.", "head_entity": "lawkholme lane", "tail_entity": "keighley", "predicted_relation": "location", "predicted_relation_id": 58}
```

```json
{"index": 3043, "sentence": "seton ' s 101 dots is the most ever allowed by temple at the liacouras center.", "head_entity": "liacouras center", "tail_entity": "temple", "predicted_relation": "operator", "predicted_relation_id": 41}
```

```json
{"index": 3056, "sentence": "scarcely one date before the coincidence did organised, unreported bayern enthusiasts painted parts of the grünwalder stadion, 1860 traditional home stadium, in the club colors of bayern, red and white.", "head_entity": "grünwalder stadion", "tail_entity": "bayern", "predicted_relation": "operator", "predicted_relation_id": 41}
```

```json
{"index": 3063, "sentence": "the preliminary limb did made under le colisée in chalon - sur - saône, at the home court of élan chalon.", "head_entity": "le colisée", "tail_entity": "élan chalon", "predicted_relation": "operator", "predicted_relation_id": 41}
```

```json
{"index": 3073, "sentence": "lingotto represents the site of the oval lingotto, the speed skating venue for the 2006 winter olympics.", "head_entity": "oval lingotto", "tail_entity": "2006 winter olympics", "predicted_relation": "participant of", "predicted_relation_id": 10}
```

```json
{"index": 3113, "sentence": "toward 1994, perrelli began sing with the well - established dansband anders engbergs.", "head_entity": "anders engbergs", "tail_entity": "dansband", "predicted_relation": "genre", "predicted_relation_id": 44}
```

```json
{"index": 3146, "sentence": "the rinspeed yello talbo makes a swiss athletes automobile produced in association with dieter meier and boris blank of the eclectic electronic group yello.", "head_entity": "boris blank", "tail_entity": "electronic", "predicted_relation": "genre", "predicted_relation_id": 44}
```

```json
{"index": 3232, "sentence": "he did mentioned in april 2007 after hyrrokkin, a giantess from norse mythology, who launched hringhorni, baldr ' s funeral ship.", "head_entity": "hringhorni", "tail_entity": "baldr", "predicted_relation": "characters", "predicted_relation_id": 42}
```

```json
{"index": 3235, "sentence": "than lima concluded down toward january 2004, the jouef label was acquired a few months later by hornby railways (tran 2004).", "head_entity": "lima", "tail_entity": "hornby railways", "predicted_relation": "owned by", "predicted_relation_id": 46}
```

```json
{"index": 3242, "sentence": "he represents developed by ivory tower and published by ubisoft for microsoft windows, playstation 4 and xbox one.", "head_entity": "ivory tower", "tail_entity": "ubisoft", "predicted_relation": "owned by", "predicted_relation_id": 46}
```

```json
{"index": 3246, "sentence": "relying to the rté bid to oireachtas commission the recommended rté nl operated saorsat makes planned to be a copying of saorview.", "head_entity": "saorview", "tail_entity": "rté", "predicted_relation": "owned by", "predicted_relation_id": 46}
```

```json
{"index": 3247, "sentence": "candy makes an aboriginal diem undertaking journal published by ht media, a delhi - based media group which also publishes \"hindustan times\".", "head_entity": "hindustan times", "tail_entity": "ht media", "predicted_relation": "owned by", "predicted_relation_id": 46}
```

```json
{"index": 3252, "sentence": "1994 victoire miinnehoma did belonged by comedian freddie starr.", "head_entity": "miinnehoma", "tail_entity": "freddie starr", "predicted_relation": "owned by", "predicted_relation_id": 46}
```

```json
{"index": 3256, "sentence": "nevertheless, on 1 june 2014 banco popolare absorbed credito bergamasco by issues 1. 412 shares of banco popolare for each shares of credito bergamasco.", "head_entity": "credito bergamasco", "tail_entity": "banco popolare", "predicted_relation": "owned by", "predicted_relation_id": 46}
```

```json
{"index": 3264, "sentence": "a 2006 er - agency of the helicopter industry in russia created oboronprom (russian helicopters) a holding company to bring together mil and manufacturing plants.", "head_entity": "russian helicopters", "tail_entity": "oboronprom", "predicted_relation": "owned by", "predicted_relation_id": 46}
```

```json
{"index": 3273, "sentence": "tonka (embracing kenner) did procured by the toy company hasbro in mid - 1991.", "head_entity": "kenner", "tail_entity": "hasbro", "predicted_relation": "owned by", "predicted_relation_id": 46}
```

```json
{"index": 3289, "sentence": "the dogg lineup was positioned as the mainstream line in chrysler ' s hierarchy, between desoto and plymouth.", "head_entity": "desoto", "tail_entity": "chrysler", "predicted_relation": "owned by", "predicted_relation_id": 46}
```

```json
{"index": 3314, "sentence": "\"vvvvvv\" did emitted on january 11, 2010 for microsoft windows and mac os x.", "head_entity": "vvvvvv", "tail_entity": "microsoft windows", "predicted_relation": "platform", "predicted_relation_id": 47}
```

```json
{"index": 3362, "sentence": "threads creek, a watercourse that comes part of the hawkesbury - nepean catchment, is located in greater western sydney, new south wales, australia.", "head_entity": "hawkesbury", "tail_entity": "nepean", "predicted_relation": "tributary", "predicted_relation_id": 48}
```

```json
{"index": 3377, "sentence": "he flows southerly across the schwarzbach, which discharges across the brend through bischofsheim its waterways rendering their trajectory throughout the franconian saale to the siphon basin of the rhine.", "head_entity": "franconian saale", "tail_entity": "brend", "predicted_relation": "tributary", "predicted_relation_id": 48}
```

```json
{"index": 3383, "sentence": "he did created because a safekeeping bastion toward the second semester from the 13th millenium near the confluence of the kysuca and the váh, where tolls were collected.", "head_entity": "váh", "tail_entity": "kysuca", "predicted_relation": "tributary", "predicted_relation_id": 48}
```

```json
{"index": 3397, "sentence": "momentous fluvial elements are the zulia, catatumbo and pamplonita rivers.", "head_entity": "catatumbo", "tail_entity": "zulia", "predicted_relation": "tributary", "predicted_relation_id": 48}
```

```json
{"index": 3412, "sentence": "13 kilograms for the nemunas nor jūra waterway confluence to the rambynas park boundary is the largest river in lithuania, the nemunas, and the lithuanian state border with russia ' s kaliningrad enclave.", "head_entity": "nemunas", "tail_entity": "jūra", "predicted_relation": "tributary", "predicted_relation_id": 48}
```

```json
{"index": 3430, "sentence": "an marvelous appearance against top - flight club arsenal in a 1950 fa cup match resulted in the londoners making an offer for the player, but swansea turned it down.", "head_entity": "1950 fa cup", "tail_entity": "arsenal", "predicted_relation": "participating team", "predicted_relation_id": 3}
```

```json
{"index": 3458, "sentence": "cska are the advocating russian premier league champions, and will take part in the russian super cup, russian cup and champions league, entering at the group stage.", "head_entity": "russian premier league", "tail_entity": "cska", "predicted_relation": "participating team", "predicted_relation_id": 3}
```

```json
{"index": 3492, "sentence": "a small mes than entering apoel, it crowned buckaroo as his team managed to win the cypriot first division title for a fourth time in the row.", "head_entity": "cypriot first division", "tail_entity": "apoel", "predicted_relation": "participating team", "predicted_relation_id": 3}
```

```json
{"index": 3496, "sentence": "german jupp heynckes makes the exclusive boy to have earned the cup twice and did so in consecutive years (2003 and 2004) with german club schalke 04.", "head_entity": "2003", "tail_entity": "schalke 04", "predicted_relation": "participating team", "predicted_relation_id": 3}
```

```json
{"index": 3508, "sentence": "sachay ’ s papa syed yarwar huseyin did meanwhile a legendary ecclesiastical poem nor a series from elegy (marsia) and nauha are to his credit.", "head_entity": "elegy", "tail_entity": "marsia", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
```

```json
{"index": 3520, "sentence": "toward vegetative biology, endoderm corresponds to the innermost part of the cortex (bark) in young shoots and young roots often consisting of a single cell layer.", "head_entity": "cortex", "tail_entity": "bark", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
```

```json
{"index": 3531, "sentence": "they did five babies: liddell (born 1879), petar (born 1881), heinz (born 1882) nor twins birthed in 1883, klaus and katharina, known as katia.", "head_entity": "katia", "tail_entity": "katharina", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
```

```json
{"index": 3552, "sentence": "variant formats of the name include kristal, krystal, cristal, krystle and kristol.", "head_entity": "kristal", "tail_entity": "krystal", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
```

```json
{"index": 3553, "sentence": "which dhul - qarnayn (normally associated with alexander the great) was said to have erected to protect people near the land of the rising sun from the savages of gog and magog.", "head_entity": "dhul-qarnayn", "tail_entity": "alexander the great", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
```

```json
{"index": 3554, "sentence": "a \"drost\" nor \"drossaard\" was a kind a bailiff in the low countries (see landdrost and seneschal).", "head_entity": "seneschal", "tail_entity": "landdrost", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
```

```json
{"index": 3560, "sentence": "variant formats of the name include kristal, krystal, cristal, krystle and kristol.", "head_entity": "cristal", "tail_entity": "krystal", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
```

```json
{"index": 3589, "sentence": "akon nor giorgio tuinfort co - produced \"mirrors\" in collaboration with kierszenbaum.", "head_entity": "mirrors", "tail_entity": "akon", "predicted_relation": "performer", "predicted_relation_id": 9}
```

```json
{"index": 3592, "sentence": "it collaborated among billy corgan during the soundtrack to the cinema \"spun\", singing the movie ' s opening song, a cover of iron maiden \"s \"number of the beast\".", "head_entity": "spun", "tail_entity": "billy corgan", "predicted_relation": "performer", "predicted_relation_id": 9}
```

```json
{"index": 3620, "sentence": "than closely six mes from reclaim, she returned to the stade (la comtesse de coigny and madelon in giordano ' s \"andrea chénier\" in seville), determined to pursue her career.", "head_entity": "andrea chénier", "tail_entity": "giordano", "predicted_relation": "composer", "predicted_relation_id": 51}
```

```json
{"index": 3622, "sentence": "palmer co - wrote the issue poem to 2013 television series \"mob city\" with mark isham entitled \"hello again\" and performed it.", "head_entity": "mob city", "tail_entity": "mark isham", "predicted_relation": "composer", "predicted_relation_id": 51}
```

```json
{"index": 3624, "sentence": "under those did the series 1 struck \"without you\", written by pete ham and tom evans of the group badfinger.", "head_entity": "without you", "tail_entity": "pete ham", "predicted_relation": "composer", "predicted_relation_id": 51}
```

```json
{"index": 3628, "sentence": "toward 2015, martinis collaborated among planter nor mj avicii to work on two new tracks for his album, \"stories\".", "head_entity": "stories", "tail_entity": "avicii", "predicted_relation": "composer", "predicted_relation_id": 51}
```

```json
{"index": 3634, "sentence": "gordine urged in moraes nor jobim throughout a recent punctuation for the film \"orfeu negro\", or \"black orpheus\" (1959).", "head_entity": "black orpheus", "tail_entity": "jobim", "predicted_relation": "composer", "predicted_relation_id": 51}
```

```json
{"index": 3715, "sentence": "the 18 - path record did emitted 20 may 2013 on wati b and on mmc, the latter founded by maître gims himself.", "head_entity": "maître gims", "tail_entity": "wati b", "predicted_relation": "record label", "predicted_relation_id": 53}
```

```json
{"index": 3811, "sentence": "the duet founded the irrespective docket label psychopathic records with alex abbiss as manager, and produced and starred in the feature films \"big money hustlas\" and \"big money rustlas\".", "head_entity": "big money rustlas", "tail_entity": "psychopathic records", "predicted_relation": "record label", "predicted_relation_id": 53}
```

```json
{"index": 3840, "sentence": "the kino was an international hit, and along with \"confession\" proved a breakthrough for anglo - amalgamated.", "head_entity": "confession", "tail_entity": "anglo-amalgamated", "predicted_relation": "record label", "predicted_relation_id": 53}
```

```json
{"index": 3889, "sentence": "maalik makes a 2016 pakistani political suspenseful film made by ashir azeem.", "head_entity": "maalik", "tail_entity": "ashir azeem", "predicted_relation": "director", "predicted_relation_id": 18}
```

```json
{"index": 3895, "sentence": "he did drafted and directed by andjar asmara, a former journalist and stage - writer who had made his directorial debut the preceding year with \"kartinah\".", "head_entity": "kartinah", "tail_entity": "andjar asmara", "predicted_relation": "director", "predicted_relation_id": 18}
```

```json
{"index": 3902, "sentence": "the roman will feature the historian from khamoshiyan nor two different tales penned by vikram bhatt – 1920 (film) and.", "head_entity": "khamoshiyan", "tail_entity": "vikram bhatt", "predicted_relation": "director", "predicted_relation_id": 18}
```

```json
{"index": 3909, "sentence": "he did adjusted for the 1962 marathi phraseology replay \"to mee navhech\", authored by pralhad keshav atre.", "head_entity": "to mee navhech", "tail_entity": "pralhad keshav atre", "predicted_relation": "director", "predicted_relation_id": 18}
```

```json
{"index": 4061, "sentence": "she went soon to the worid cup for the 2015 – 16 season, taking 15th place at the world championships in igls and 11th in the season rankings.", "head_entity": "world championships", "tail_entity": "igls", "predicted_relation": "location", "predicted_relation_id": 58}
```

```json
{"index": 4069, "sentence": "this did the fourth yu. s. open through the lake trajectory of the olympic club; the first two in 1955 and 1966 ended in playoffs, and 1987 was a one - stroke victory.", "head_entity": "1987", "tail_entity": "olympic club", "predicted_relation": "location", "predicted_relation_id": 58}
```

```json
{"index": 4073, "sentence": "the camshaft makes yet displayed annually in the tradition of pittsburgh ' s light up night at the horne ' s building.", "head_entity": "light up night", "tail_entity": "pittsburgh", "predicted_relation": "location", "predicted_relation_id": 58}
```

```json
{"index": 4095, "sentence": "during his birthplace hedemora one of the buildings he designed, the wahlman building, bears his name.", "head_entity": "wahlman building", "tail_entity": "hedemora", "predicted_relation": "location", "predicted_relation_id": 58}
```

```json
{"index": 4103, "sentence": "apostrophe leads regular fieldwork activities in the canadian rockies and discovered an important new burgess shale site near marble canyon in kootenay national park in 2012.", "head_entity": "burgess shale", "tail_entity": "kootenay national park", "predicted_relation": "location", "predicted_relation_id": 58}
```

```json
{"index": 4105, "sentence": "toward rapid 2013, the colorado parliament voted a suite from gun audit billings following the teatro shooting in aurora, colorado and the sandy hook school shooting in newtown, connecticut.", "head_entity": "sandy hook school shooting", "tail_entity": "newtown, connecticut", "predicted_relation": "location", "predicted_relation_id": 58}
```

```json
{"index": 4179, "sentence": "this quantity was first proposed in the 1990s by british anthropologist robin dunbar, who found a correlation between primate brain size and average social group size.", "head_entity": "robin dunbar", "tail_entity": "anthropologist", "predicted_relation": "occupation", "predicted_relation_id": 36}
```

```json
{"index": 4180, "sentence": "throughout illustration, soviet algebra nor economist leonid kantorovich worded the sprouts from linear programming for 1939 through efforts to solve a plywood manufacturer ' s production optimization issues.", "head_entity": "leonid kantorovich", "tail_entity": "linear programming", "predicted_relation": "field of work", "predicted_relation_id": 59}
```

```json
{"index": 4216, "sentence": "the cause from the assassinate was revenge for markelov ' s prior work as a lawyer in the interests of anti - russian activists.", "head_entity": "markelov", "tail_entity": "russian", "predicted_relation": "country of citizenship", "predicted_relation_id": 6}
```

```json
{"index": 4226, "sentence": "one from the most impenetrable qualities from the \"hildebrandslied\" makes its lexicon, which is a mixture of old high german (with some specifically bavarian features) and old saxon.", "head_entity": "hildebrandslied", "tail_entity": "old high german", "predicted_relation": "language of work or name", "predicted_relation_id": 60}
```

```json
{"index": 4350, "sentence": "for the time of the occupy wall street protests, white was working as a senior editor for \"adbusters\" magazine with occupy wall street co - founder kalle lasn.", "head_entity": "kalle lasn", "tail_entity": "adbusters", "predicted_relation": "member of", "predicted_relation_id": 77}
```

所有样本已分析完毕。每个样本的输出格式均按照要求，包含index、sentence、head_entity、tail_entity、predicted_relation和predicted_relation_id字段。