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
  "index": 样本的index,
  "sentence": 还原的原句,
  "head_entity": 头实体,
  "tail_entity": 尾实体,
  "predicted_relation": 预测的关系名称,
  "predicted_relation_id": 对应的关系ID
}
```

请上传您的txt文件，我会立即开始处理。

---
# 2、1.txtTXT 22.97K...
**问题详情：**
1.txtTXT 22.97KB请持续输出不要停止直到分析完每一个样本，注意不要漏掉样本

---
**回答（Answer）：**
我将按照要求分析每个样本，输出完整的JSON格式结果。让我们开始：

```json
{"index": 4411, "sentence": "In 1979, he and Kay (forthwith his marchioness) opened \"K-Paul's Louisiana Kitchen\" in the French Quarter of New Orleans.", "head_entity": "french quarter", "tail_entity": "louisiana", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 4431, "sentence": "The Senne Railway () is a singular-runway subdivision line from Brackwede to Paderborn with a through service to Bielefeld in the German state of North Rhine-Westphalia.", "head_entity": "brackwede", "tail_entity": "bielefeld", "predicted_relation": "part of", "predicted_relation_id": 67}
```

```json
{"index": 4434, "sentence": "ICGA is coming up on a campus near Valapattanam firth on the Arabian Sea in Irinavu near Azhikkal in Kannur district of yankee Kerala.", "head_entity": "valapattanam", "tail_entity": "kannur district", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 4446, "sentence": "wore in Palamós, Girona, Catalonia, Condom was a youth product at FC Barcelona's La Masia.", "head_entity": "girona", "tail_entity": "catalonia", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 4448, "sentence": "The Cuvilliés Theatre () or Old Residence Theatre (\"Altes Residenztheater\") is the former cloister theatre of the Residenz in Munich, Bavaria, Germany.", "head_entity": "residence theatre", "tail_entity": "munich", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 4455, "sentence": "She was firstly reelected to the Hawaii House of Representatives in 2004 profile the Kāhala country of Honolulu on Oahu.", "head_entity": "kāhala", "tail_entity": "honolulu", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 4457, "sentence": "Alexander Kuprin perished on 25 August 1938, and was interred in Volkovo Cemetery's Literaturskiye Mostki (Literary Bridge) in Leningrad two days later.", "head_entity": "volkovo cemetery", "tail_entity": "leningrad", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 4461, "sentence": "presently, the most measurable dromes are in Lisbon, Porto, Faro, Funchal (Madeira) and Ponta Delgada (Azores).", "head_entity": "funchal", "tail_entity": "madeira", "predicted_relation": "part of", "predicted_relation_id": 67}
```

```json
{"index": 4471, "sentence": "Scott is placed in north Brown County along the southeast shore of Green Bay, an arm of Lake Michigan.", "head_entity": "green bay", "tail_entity": "brown county", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 4580, "sentence": "withal, it has been referenced and parodied by some Korean sampler fairs, including SBS's \"X-Man\" and KBS's \"Star Golden Bell\".", "head_entity": "star golden bell", "tail_entity": "korean", "predicted_relation": "country of origin", "predicted_relation_id": 16}
```

```json
{"index": 4583, "sentence": "It was published on DVD by Vanguard Cinema in May 2005 as \"Si Te dicen que caí\" (\"If They Tell you I Fell\"), in Spanish with English captions.", "head_entity": "if they tell you i fell", "tail_entity": "spanish", "predicted_relation": "original language of film or TV show", "predicted_relation_id": 65}
```

```json
{"index": 4596, "sentence": "She besides voiced Chihiro's mother in the original Japanese version of \"Spirited Away\".", "head_entity": "spirited away", "tail_entity": "japanese", "predicted_relation": "original language of film or TV show", "predicted_relation_id": 65}
```

```json
{"index": 4598, "sentence": "In November 2006, he learnt part in the singing competition \"Fame X\" (in Hindi language \"Chal Udiye\") inaugural 2006-2007 season.", "head_entity": "fame x", "tail_entity": "hindi language", "predicted_relation": "original language of film or TV show", "predicted_relation_id": 65}
```

```json
{"index": 4608, "sentence": "In 2010 he appears as a node artist in the Afrikaans improvisational TV-show Proesstraat and as judge on the popular Afrikaans reality show Supersterre alongside Shaleen Surtie-Richards and Theuns Jordaan.", "head_entity": "proesstraat", "tail_entity": "afrikaans", "predicted_relation": "original language of film or TV show", "predicted_relation_id": 65}
```

```json
{"index": 4704, "sentence": "In generative phonology, another soma of transformation is the phonological rule, which describes a mapping between an underlying representation (the phoneme) and the surface form that is articulated during natural speech.", "head_entity": "phonological rule", "tail_entity": "generative phonology", "predicted_relation": "part of", "predicted_relation_id": 67}
```

```json
{"index": 4707, "sentence": "Carr, who had embarked off with a major LP against Cooper's Ohio State, seemed to have much more trouble beating Tressel's version of the Buckeyes.", "head_entity": "ohio state", "tail_entity": "buckeyes", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
```

```json
{"index": 4708, "sentence": "It was too the paster to be qualified for assign together the soundtrack to the Eminem-starring film \"8 Mile\", which had the lead single \"Lose Yourself\".", "head_entity": "lose yourself", "tail_entity": "the soundtrack", "predicted_relation": "part of", "predicted_relation_id": 67}
```

```json
{"index": 4735, "sentence": "Hoch made her WTA go debut at the 2007 Gastein Ladies in both singles and doubles.", "head_entity": "singles", "tail_entity": "2007 gastein ladies", "predicted_relation": "competition class", "predicted_relation_id": 66}
```

```json
{"index": 4752, "sentence": "Nightingale Island is voice of the Nightingale Islands, which also includes islets Middle Island and Stoltenhoff Island.", "head_entity": "stoltenhoff island", "tail_entity": "nightingale islands", "predicted_relation": "part of", "predicted_relation_id": 67}
```

```json
{"index": 4757, "sentence": "Canada continued to boom across North America as other British colonies and territories joined with or were ceded to Canada, eventually growing from four provinces to ten provinces and three territories.", "head_entity": "colonies", "tail_entity": "north america", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
```

```json
{"index": 4808, "sentence": "In 1992, with grand hymeneals, the Orioles began their season in a brand new ballpark, Oriole Park at Camden Yards, and thus retiring Memorial Stadium in the major league baseball world.", "head_entity": "memorial stadium", "tail_entity": "baseball", "predicted_relation": "instance of", "predicted_relation_id": 28}
```

```json
{"index": 4974, "sentence": "For over 700 dayss, Örebro Castle has kept a watchful eye on everyone crossing the bridge on the River Svartån.", "head_entity": "örebro castle", "tail_entity": "svartån", "predicted_relation": "located in or next to body of water", "predicted_relation_id": 71}
```

```json
{"index": 4989, "sentence": "The additional streamliners in the peak hr between Essen and Haltern also bear this name, although this is not geographically correct as the Niers flows through Mönchengladbach.", "head_entity": "mönchengladbach", "tail_entity": "niers", "predicted_relation": "crosses", "predicted_relation_id": 64}
```

```json
{"index": 4991, "sentence": "The geographic co-ordinates of Firstview are roughly the antipodes of the coordinates of Île Saint-Paul, a French island in the Indian Ocean.", "head_entity": "île saint-paul", "tail_entity": "indian ocean", "predicted_relation": "located in or next to body of water", "predicted_relation_id": 71}
```

```json
{"index": 5123, "sentence": "Windows Vista, Windows 7, Windows Server 2008 and 2008 R2 too tolerates IPP printing over RPC in the \"Medium-Low\" security zone.", "head_entity": "2008 r2", "tail_entity": "windows server 2008", "predicted_relation": "followed by", "predicted_relation_id": 26}
```

```json
{"index": 5173, "sentence": "Artoustes were licence-cantilevered by Bristol Siddeley (formerly Blackburn) in the UK, Hindustan Aeronautics Limited in India, and developed by Continental CAE in the USA as the Continental T51.", "head_entity": "bristol siddeley", "tail_entity": "blackburn", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
```

```json
{"index": 5174, "sentence": "Two motorcars were used for the vocal editing in songs like \"Hollywood\" and \"Nobody Knows Me\".", "head_entity": "nobody knows me", "tail_entity": "hollywood", "predicted_relation": "part of", "predicted_relation_id": 67}
```

```json
{"index": 5183, "sentence": "Theodora Komnene, grandniece of Manuel and Queen choir of Baldwin III of Jerusalem, was cardinal geezerhoods darkened at her matrimony in 1158.", "head_entity": "theodora komnene", "tail_entity": "baldwin iii of jerusalem", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5186, "sentence": "Birgitte foremost constellated Prince Richard of Gloucester, the junior boy of Prince Henry, Duke of Gloucester, and Princess Alice, Duchess of Gloucester, in the late 1960s in Cambridge.", "head_entity": "prince henry, duke of gloucester", "tail_entity": "princess alice, duchess of gloucester", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5190, "sentence": "Joel Keller of telecasting Squad said the no-dating policy felt like a plot contrivance meant strictly to keep Ben and Leslie apart, which he felt was not good for the story.", "head_entity": "ben", "tail_entity": "leslie", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5193, "sentence": "She was betrothed to Henry Brandon, 1st Earl of Lincoln, the Word of Charles Brandon, 1st Duke of Suffolk, and Princess Mary, the sister of Henry VIII.", "head_entity": "charles brandon, 1st duke of suffolk", "tail_entity": "princess mary", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5218, "sentence": "Atia Balba Caesonia (hatched 85 BC) was the second girl of Julius Caesar's younger sister, Julia Minor and Marcus Atius Balbus, son of a Senator from Aricia.", "head_entity": "marcus atius balbus", "tail_entity": "julia minor", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5220, "sentence": "Shah is the junior Word of actors Naseeruddin Shah and Ratna Pathak.", "head_entity": "naseeruddin shah", "tail_entity": "ratna pathak", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5222, "sentence": "David Ackroyd and Joan Van Ark appeared as Lucy's begetters Gary and Valene Ewing.", "head_entity": "valene ewing", "tail_entity": "gary", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5225, "sentence": "This was the muse of the torment that stalked Hippodamia and Pelops' children Atreus and Thyestes as well as their descendents Agamemnon, Aegisthus, Menelaus and Orestes.", "head_entity": "hippodamia", "tail_entity": "pelops", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5230, "sentence": "It besides features their fathers Kris and Caitlyn Jenner (previously known as Bruce Jenner), and brother Rob Kardashian.", "head_entity": "kris", "tail_entity": "caitlyn jenner", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5234, "sentence": "\"; at ancestry.com in Merseyside, United Kingdom, the single kid of actors Annette Ekblom and Alun Lewis.", "head_entity": "annette ekblom", "tail_entity": "alun lewis", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5239, "sentence": "He was the boy of dancers Pandit Uday Shankar and Amala Shankar and the nephew of the sitar maestro Ravi Shankar.", "head_entity": "amala shankar", "tail_entity": "uday shankar", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5248, "sentence": "During a sojourn to the apartment of Sonic Youth members Thurston Moore and Kim Gordon, Moore told Vig he wanted the album to sound like an obscure Mecht Mensch single that Vig had produced.", "head_entity": "kim gordon", "tail_entity": "thurston moore", "predicted_relation": "spouse", "predicted_relation_id": 74}
```

```json
{"index": 5324, "sentence": "In both the \"Poetic Edda\" and the \"Prose Edda\", Gerðr is wrote as the girl of Gymir and the jötunn Aurboða.", "head_entity": "gerðr", "tail_entity": "aurboða", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5328, "sentence": "Rai misdated Aimee Osbourne, girl of rocker Ozzy Osbourne and celebrity personality Sharon Osbourne and sister to Kelly and Jack Osbourne.", "head_entity": "jack osbourne", "tail_entity": "sharon osbourne", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5332, "sentence": "stalk Ögedei Khan's grave, Töregene becomes the Regent of the Mongol Empire for some years until her Junior, Güyük, is returned as the new Great Khan.", "head_entity": "güyük", "tail_entity": "töregene", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5333, "sentence": "Vibia Aurelia Sabina (170-died before 217), girl and least kid of Roman Emperor Marcus Aurelius and Roman Empress Faustina the Younger was a great, great niece to Vibia Sabina.", "head_entity": "vibia aurelia sabina", "tail_entity": "faustina the younger", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5344, "sentence": "Shapur IV was the Word of Yazdegerd I and Shushandukht, and had two brothers named Bahram V and Narse.", "head_entity": "bahram v", "tail_entity": "shushandukht", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5348, "sentence": "An cyclic fairytale is that after Dushyanta fluffed to discriminate Shakuntala, her mummy Menaka took Shakuntala to Heaven where she gave birth to Bharata.", "head_entity": "shakuntala", "tail_entity": "menaka", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5353, "sentence": "After the sphacelus of Alghu in 1266, Orghana appointed her Word Mubarak Shah by her first husband, Qara Hulagu as khan of the Chagatai Khanate.", "head_entity": "mubarak shah", "tail_entity": "orghana", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5355, "sentence": "Dyah Gitarja was the girl of Raden Wijaya, the first king of Majapahit, and his consort Dyah Gayatri Rajapatni.", "head_entity": "dyah gitarja", "tail_entity": "gayatri rajapatni", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5360, "sentence": "The lettering on the Cairo chip accounts Khenthap as the ma of king Djer. Aidan Dodson & Dyan Hilton: \"The Complete Royal Families of Ancient Egypt\".", "head_entity": "djer", "tail_entity": "khenthap", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5362, "sentence": "The girl of starlet Polly Adams and actor Richard Owens, she is the sister of fellow actress Caroline Harker.", "head_entity": "caroline harker", "tail_entity": "polly adams", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5365, "sentence": "Nicolson was the senior Junior of writers Harold Nicolson and Vita Sackville-West and the brother of writer and politician Nigel.", "head_entity": "nigel", "tail_entity": "vita sackville-west", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5366, "sentence": "Swatilekha Sengupta's girl Sohini Sengupta would appear in a guest appearance.", "head_entity": "sohini sengupta", "tail_entity": "swatilekha sengupta", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5370, "sentence": "moreover, she backs her mother Arlyn Phoenix in the charity organisation \"River Phoenix Centre for Peacebuilding\", where she is the office manager.", "head_entity": "river phoenix", "tail_entity": "arlyn phoenix", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5372, "sentence": "Lepidus and Junia Secunda had at least one shaver, Marcus Aemilius Lepidus the Younger.", "head_entity": "marcus aemilius lepidus the younger", "tail_entity": "junia secunda", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5382, "sentence": "Vladimir and Rogneda had five shavers and the firstest of them, Izyaslav, became Prince of Polotsk (ruled 989-1001).", "head_entity": "izyaslav", "tail_entity": "rogneda", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5389, "sentence": "Leah responds by subscribe her servant Zilpah to Jacob, and titles and lifts the two sons (Gad and Asher) that Zilpah bears.", "head_entity": "gad", "tail_entity": "zilpah", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5395, "sentence": "\"In Your centers\" is a ballade debuted by Onew of son citizenry Shinee.", "head_entity": "onew", "tail_entity": "shinee", "predicted_relation": "member of", "predicted_relation_id": 77}
```

```json
{"index": 5415, "sentence": "In betimes June 2006, it was advertized through the German \"Rock Hard\" cartridge that Abbath and Horgh would reunify as Immortal.", "head_entity": "abbath", "tail_entity": "immortal", "predicted_relation": "member of", "predicted_relation_id": 77}
```

```json
{"index": 5417, "sentence": "Thomas Bushnell, BSG, once known as Michael Bushnell (born 1967), is a software developer and Gregorian friar.", "head_entity": "michael bushnell", "tail_entity": "bsg", "predicted_relation": "member of", "predicted_relation_id": 77}
```

```json
{"index": 5438, "sentence": "On June 22, Kamelot acquainted Seventh Wonder's Tommy Karevik as their new official vocalist.", "head_entity": "tommy karevik", "tail_entity": "kamelot", "predicted_relation": "member of", "predicted_relation_id": 77}
```

```json
{"index": 5466, "sentence": "Shulze was the grandson of Henry Muhlenberg and the grandnephew of brethren Peter Muhlenberg and Frederick Muhlenberg, who were leading politicians.", "head_entity": "henry muhlenberg", "tail_entity": "frederick muhlenberg", "predicted_relation": "sibling", "predicted_relation_id": 33}
```

```json
{"index": 5469, "sentence": "In Greek mythology, Doryclus is a Word of king Priam mentioned in Homer's \"Iliad\", Hyginus's Fabulae and the \"Bibliotheca\".", "head_entity": "priam", "tail_entity": "doryclus", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 5472, "sentence": "In \"Lebor Gabála Érenn\", Badb, Macha and Morrígan make up the Morrígna triad and are named as daughters of the farming goddess Ernmas.", "head_entity": "ernmas", "tail_entity": "the morrígna", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5475, "sentence": "The freshman holder of the Fraulein was Wakda (瓦克達; died 1652), Daišan's fourth son and a grandson of Nurhaci (the founder of the Qing dynasty).", "head_entity": "nurhaci", "tail_entity": "daišan", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 5477, "sentence": "Operating from the former domicile of former President of South Africa Thabo Mbeki, the deftness was named after Linda Mbeki, the daughter of Govan and Epainette Mbeki following her death in 2005.", "head_entity": "govan", "tail_entity": "thabo mbeki", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 5479, "sentence": "After thrashing by Shahu, Sambhaji's stepmother, Tarabai so localized up a rival Court in Kolhapur with her Word Shivaji II as Raja of Kolhapur from 1710 to 1714.", "head_entity": "tarabai", "tail_entity": "shivaji ii", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5480, "sentence": "harmonize to Karel van Mander, he was the second son of Pieter Aertsen, the younger brother of Pieter Pietersz the Elder, and the older brother of Dirck Pietersz.", "head_entity": "pieter aertsen", "tail_entity": "pieter pietersz the elder", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 5484, "sentence": "John Strasberg (twinned May 20, 1941 in New York City) is the Word of Lee and Paula Strasberg of the Actors Studio, and brother of actress Susan Strasberg.", "head_entity": "paula strasberg", "tail_entity": "susan strasberg", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5485, "sentence": "During an examination of neglect babies, Damian Wayne, AKA Robin, Batman's Word with Talia al Ghul, has taken the Batmobile out to an abandoned toy factory without Batman's consent.", "head_entity": "talia al ghul", "tail_entity": "damian wayne", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5486, "sentence": "He is the Word of Gordie and Colleen Howe, younger brother of Marty Howe, and nephew of Vic Howe.", "head_entity": "colleen howe", "tail_entity": "marty howe", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5487, "sentence": "His girls, by Ernmas, were the three eponymous Irish goddesses Ériu, Banba, and Fodla.", "head_entity": "ernmas", "tail_entity": "ériu", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5491, "sentence": "He is a boy of Enlil and Ninlil, along with Nanna and Ninurta.", "head_entity": "ninlil", "tail_entity": "nanna", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5506, "sentence": "The Naomi adverted in the straight refers to Naomi Foner, who was a manufacturer on the display during the basic two Christmastides and is the ma of actress Maggie Gyllenhaal and actor Jake Gyllenhaal.", "head_entity": "naomi foner", "tail_entity": "maggie gyllenhaal", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5507, "sentence": "Trampoline's second filly, Glencoe (by Sultan), was peradventure her most known Jnr.", "head_entity": "sultan", "tail_entity": "glencoe", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 5508, "sentence": "On the sphacelus of Yesugei, Temujin, his ma Hoelun, his siblings and two half-brothers (including Behter) were abandoned by their tribe and left to fend for themselves.", "head_entity": "hoelun", "tail_entity": "temujin", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5516, "sentence": "\"Astronomica Book 2.12.1\" Pierus, the begetter of Hyacinth by the Muse Clio, was also called a son of Magnes. Apollodorus\".", "head_entity": "pierus", "tail_entity": "hyacinth", "predicted_relation": "father", "predicted_relation_id": 19}
```

```json
{"index": 5523, "sentence": "Lionel Barrymore was birthed Lionel Herbert Blythe in Philadelphia, the Junior of actors Georgiana Drew Barrymore and Maurice Barrymore (born Herbert Arthur Chamberlayne Blyth).", "head_entity": "georgiana drew", "tail_entity": "lionel barrymore", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5527, "sentence": "Three shipwrecks of the Royal Norwegian Navy have abode the hypocorism HNoMS \"Vale\", after Váli, Word of the god Odin and the giantess Rindr:", "head_entity": "rindr", "tail_entity": "váli", "predicted_relation": "mother", "predicted_relation_id": 76}
```

```json
{"index": 5570, "sentence": "He is depicted in the Jackie Robinson biopic \"42\" as one of Robinson's earliest supporters.", "head_entity": "\"42\"", "tail_entity": "jackie robinson", "predicted_relation": "main subject", "predicted_relation_id": 79}
```

```json
{"index": 5581, "sentence": "The brothel is burst prominently in The Negotiator (1998) where it serves as the location of offices for the Internal Affairs Division of the Chicago Police Department.", "head_entity": "the negotiator", "tail_entity": "chicago police department", "predicted_relation": "work location", "predicted_relation_id": 38}
```

```json
{"index": 5584, "sentence": "In 1628, Cornelius Jansen, a prof at Leuven, recommenced hyphenate \"Augustinus\", a three crescendo dissertation on Augustine of Hippo which conflated Jesuits with Pelagianism by highlighting Augustine's propositions.", "head_entity": "augustinus", "tail_entity": "pelagianism", "predicted_relation": "notable work", "predicted_relation_id": 62}
```

所有样本分析完毕。