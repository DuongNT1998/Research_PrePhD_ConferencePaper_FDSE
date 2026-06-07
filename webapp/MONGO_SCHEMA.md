# MongoDB schema (`rl_chat` database)

Three collections. IDs are Mongo `ObjectId`; timestamps are UTC datetimes.

## `users`
```jsonc
{
  "_id": ObjectId,
  "email": "a@b.com",          // unique index
  "username": "alice",
  "password_hash": "<bcrypt>", // never returned by the API
  "created_at": ISODate
}
```
Indexes: `email` (unique), `username`.

## `threads`
```jsonc
{
  "_id": ObjectId,
  "user_id": ObjectId,         // owner (ref users._id)
  "title": "lightweight gaming laptop…",  // auto-generated from first query
  "created_at": ISODate,
  "updated_at": ISODate        // bumped on each new message
}
```
Indexes: `user_id`.

## `messages`
```jsonc
{
  "_id": ObjectId,
  "thread_id": ObjectId,       // ref threads._id
  "user_id": ObjectId,
  "role": "user" | "assistant",
  "content": "…",
  "meta": {                    // populated for assistant messages only
    "hops": 3,
    "confidence": 0.87,
    "reasoning_path": ["Product -[...]-> Aspect(battery)", "…"],
    "evidence": ["…", "…"]
  },
  "created_at": ISODate
}
```
Indexes: `thread_id`, compound `(thread_id, created_at)`.
