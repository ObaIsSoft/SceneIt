"""Find trailers on YouTube and add them to the library.

Replaces build_trailer_database.py, which carried a CURATED_TRAILERS list of 15
hardcoded YouTube URLs, merged them with API results, and wrote the outcome to
sceneit_trailers.json for config.py to read back at import time. Discovery
results now go straight into the library as rows.

Nothing here names a film. What gets indexed is decided by the search terms you
pass, or by adding sources directly with `python app.py --add <url>`.

    python discover.py "official movie trailer 2026" --limit 5
    python discover.py --query-file queries.txt --dry-run
"""

import argparse
import sys

import config
import library


def search_youtube(query, limit):
    """Return [{title, url, ...}] for one query, or [] if unavailable."""
    if not config.YOUTUBE_API_KEY:
        print("No YOUTUBE_API_KEY set. Put one in .env, or add sources directly:")
        print("    python app.py --add 'https://...' --title 'Some Film'")
        return []

    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        print("google-api-python-client is missing. Install with:")
        print("    pip install -e '.[ingest]'")
        return []

    try:
        youtube = build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY)
        response = (
            youtube.search()
            .list(
                q=query,
                part="id,snippet",
                type="video",
                maxResults=min(limit, 50),
                videoDuration="short",      # trailers, not full uploads
                videoDefinition="high",
                relevanceLanguage="en",
                safeSearch="moderate",
            )
            .execute()
        )
    except HttpError as e:
        print(f"YouTube API error: {e}")
        return []
    except Exception as e:
        print(f"YouTube search failed: {e}")
        return []

    results = []
    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        video_id = (item.get("id") or {}).get("videoId")
        if not video_id:
            continue
        results.append({
            "title": snippet.get("title", "").strip(),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": snippet.get("channelTitle", ""),
            "published": (snippet.get("publishedAt") or "")[:10],
            "description": (snippet.get("description") or "").strip(),
        })
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="search terms")
    parser.add_argument("--query-file", help="file with one query per line")
    parser.add_argument("--limit", type=int, default=10, help="results per query")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be added, change nothing")
    args = parser.parse_args(argv)

    queries = list(args.query)
    if args.query_file:
        with open(args.query_file) as fh:
            queries += [line.strip() for line in fh if line.strip()]
    if not queries:
        parser.error("give at least one query, or --query-file")

    from database import init_db, migrate
    init_db()
    migrate()

    seen, added, skipped = set(), 0, 0
    for query in queries:
        print(f"\nSearching: {query}")
        for item in search_youtube(query, args.limit):
            if item["url"] in seen:
                continue
            seen.add(item["url"])

            if args.dry_run:
                print(f"  would add: {item['title'][:70]}")
                added += 1
                continue

            row, created = library.add_source(
                item["url"],
                title=item["title"] or None,
                synopsis=item["description"][:500] or None,
                metadata={"channel": item["channel"], "published": item["published"]},
            )
            if created:
                added += 1
                print(f"  added  [{row['id']}] {row['title'][:66]}")
            else:
                skipped += 1
                print(f"  exists [{row['id']}] {row['title'][:66]}")

    print(f"\n{added} added, {skipped} already present.")
    if added and not args.dry_run:
        print("Next: python app.py --extract --pending")
    return 0


if __name__ == "__main__":
    sys.exit(main())
