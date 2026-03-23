// tokenize_stem_demo.cpp
// Minimal C++ demo: tokenize → stopword filter → stem (Snowball) → toy lemmatize.
// Need to install snowball
/* % g++ -std=c++17 -O2 tokenize_stem_demo.cpp -o nlp_demo \
-I"$(brew --prefix)/include" \
-L"$(brew --prefix)/lib" -lstemmer
./ nlp_demo

Output:
Tokens:   processing is essential in natural language processing tokenization stop words stemming and lemmatization
Filtered: processing essential natural language processing tokenization stop words stemming lemmatization
Stems:    process essenti natur languag process token stop word stem lemmat
Lemmas:  process essential natural language process tokenization stop word stemm lemmatization

* We are using a toy lemmatizer for illustration; use UDPipe for real lemmatization.

*/

#include <libstemmer.h>
#include <algorithm>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

	// Lowercase helper
	static std::string
	toLower(std::string s)
{
	std::transform(s.begin(), s.end(), s.begin(), ::tolower);
	return s;
}

// 1) Tokenize (simple regex word tokenizer; keeps apostrophes inside words)
std::vector<std::string> tokenizeText(const std::string &text)
{
	static const std::regex rx(R"([A-Za-z]+(?:'[A-Za-z]+)?)");
	std::sregex_iterator it(text.begin(), text.end(), rx), end;
	std::vector<std::string> out;
	for (; it != end; ++it)
		out.push_back(toLower(it->str()));
	return out;
}

// 2) Stop-word removal (tiny demo list; extend for production)
std::vector<std::string> removeStopWords(const std::vector<std::string> &toks)
{
	static const std::unordered_set<std::string> stop = {
		"a", "an", "the", "and", "or", "but", "if", "in", "on", "of", "to", "for",
		"is", "are", "was", "were", "with", "as", "by", "at"};
	std::vector<std::string> out;
	out.reserve(toks.size());
	for (const auto &w : toks)
		if (!stop.count(w))
			out.push_back(w);
	return out;
}

// 3) Stemming via libstemmer (Porter2)
std::vector<std::string> stemTokens(const std::vector<std::string> &toks)
{
	std::vector<std::string> out;
	out.reserve(toks.size());
	sb_stemmer *st = sb_stemmer_new("english", nullptr);
	for (const auto &w : toks)
	{
		const sb_symbol *s = sb_stemmer_stem(st, (const sb_symbol *)w.data(), (int)w.size());
		out.emplace_back((const char *)s, sb_stemmer_length(st));
	}
	sb_stemmer_delete(st);
	return out;
}

// 4) Very small illustrative lemmatizer (NOT a full lemmatizer)
static std::string stripSuffix(std::string w, const std::string &suf)
{
	if (w.size() > suf.size() && w.rfind(suf) == w.size() - suf.size())
		w.erase(w.size() - suf.size());
	return w;
}
std::vector<std::string> lemmatizeTokens(const std::vector<std::string> &toks)
{
	std::vector<std::string> out;
	out.reserve(toks.size());
	for (auto w : toks)
	{
		if (w.size() > 4 && w.rfind("ies") == w.size() - 3)
			w.replace(w.size() - 3, 3, "y");
		else if (w.size() > 3 && w.back() == 's' && !(w.size() > 2 && (w.substr(w.size() - 2) == "ss" || w.substr(w.size() - 2) == "us")))
			w.pop_back();
		if (w.size() > 5 && w.rfind("ing") == w.size() - 3)
			w = stripSuffix(w, "ing");
		if (w.size() > 4 && w.rfind("ed") == w.size() - 2)
			w = stripSuffix(w, "ed");
		out.push_back(w);
	}
	return out;
}

int main()
{
	std::string text =
		"Processing is essential in Natural Language Processing—tokenization, stop-words, stemming, and lemmatization.";

	auto tokens = tokenizeText(text);
	auto filtered = removeStopWords(tokens);
	auto stems = stemTokens(filtered);
	auto lemmas = lemmatizeTokens(filtered); // illustrative only

	std::cout << "Tokens:   ";
	for (auto &w : tokens)
		std::cout << w << ' ';
	std::cout << '\n';
	std::cout << "Filtered: ";
	for (auto &w : filtered)
		std::cout << w << ' ';
	std::cout << '\n';
	std::cout << "Stems:    ";
	for (auto &w : stems)
		std::cout << w << ' ';
	std::cout << '\n';
	std::cout << "Lemmas:  ";
	for (auto &w : lemmas)
		std::cout << w << ' ';
	return 0;
}
