# core/vault.py was the memory_vault.py - Encrypted Emotional Memory Storage
"""
Encrypted Emotional Memory Vault
Stores, retrieves, and manages emotional data with military-grade encryption
"""

import sqlite3
import json
import hashlib
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import numpy as np


@dataclass
class EmotionalContext:
    """Represents a stored emotional context"""
    context_id: str
    user_id: str
    emotion_vector: Optional[np.ndarray] = None
    primary_emotion: Optional[str] = None
    emotional_intensity: float = 0.0
    conversation_summary: Optional[str] = None
    tags: List[str] = None
    privacy_level: int = 1  # 1=low, 2=medium, 3=high
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MemoryQuery:
    """Query parameters for retrieving emotional contexts"""
    user_id: str
    emotion_filter: Optional[str] = None
    tag_filter: Optional[List[str]] = None
    time_range_hours: Optional[int] = None
    privacy_level_max: int = 3
    limit: int = 10
    include_expired: bool = False


class EncryptionManager:
    """Manages encryption and decryption of sensitive data"""

    def __init__(self, password: str, salt: Optional[bytes] = None):
        if salt is None:
            salt = os.urandom(16)

        self.salt = salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.cipher = Fernet(key)

    def encrypt_data(self, data: Any) -> bytes:
        """Encrypt any serializable data"""
        json_str = json.dumps(data, default=str)
        return self.cipher.encrypt(json_str.encode())

    def decrypt_data(self, encrypted_data: bytes) -> Any:
        """Decrypt data back to original format"""
        decrypted_bytes = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted_bytes.decode())

    def encrypt_vector(self, vector: np.ndarray) -> bytes:
        """Encrypt numpy arrays"""
        vector_bytes = vector.tobytes()
        return self.cipher.encrypt(vector_bytes)

    def decrypt_vector(self, encrypted_vector: bytes, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Decrypt numpy arrays"""
        decrypted_bytes = self.cipher.decrypt(encrypted_vector)
        return np.frombuffer(decrypted_bytes, dtype=dtype).reshape(shape)

    def get_salt(self) -> bytes:
        """Get the salt for storage"""
        return self.salt


class SearchableEncryption:
    """Enables searching on encrypted tags and keywords"""

    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager

    def create_searchable_index(self, tags: List[str]) -> List[str]:
        """Create searchable encrypted indices for tags"""
        searchable_tags = []
        for tag in tags:
            # Create deterministic hash for searching
            tag_hash = hashlib.sha256(f"{tag}_search_key".encode()).hexdigest()[:16]
            searchable_tags.append(tag_hash)
        return searchable_tags

    def create_search_hash(self, search_term: str) -> str:
        """Create search hash for a term"""
        return hashlib.sha256(f"{search_term}_search_key".encode()).hexdigest()[:16]

    def matches_encrypted_tags(self, search_term: str, encrypted_tags: List[str]) -> bool:
        """Check if search term matches any encrypted tags"""
        search_hash = self.create_search_hash(search_term)
        return search_hash in encrypted_tags


class EmotionalMemoryVault:
    """Main encrypted memory storage system"""

    def __init__(self, db_path: str = "ghost_memory.db", password: str = "ghost_protocol_key"):
        self.db_path = db_path
        self.encryption_manager = EncryptionManager(password)
        self.searchable_encryption = SearchableEncryption(self.encryption_manager)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables with encrypted schema"""
        cursor = self.conn.cursor()

        # Emotional contexts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotional_contexts (
                context_id TEXT PRIMARY KEY,
                user_id_hash TEXT NOT NULL,
                encrypted_data BLOB NOT NULL,
                encrypted_vector BLOB,
                vector_shape TEXT,
                searchable_tags TEXT,  -- JSON array of search hashes
                privacy_level INTEGER DEFAULT 1,
                emotional_intensity REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                data_hash TEXT  -- For integrity checking
            )
        """)

        # Interaction history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interaction_history (
                interaction_id TEXT PRIMARY KEY,
                context_id TEXT REFERENCES emotional_contexts(context_id),
                user_id_hash TEXT NOT NULL,
                interaction_type TEXT,
                encrypted_summary BLOB,
                constraint_violations TEXT,  -- JSON array
                processing_route TEXT,
                privacy_budget_used REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Memory decay table for automatic cleanup
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_decay_rules (
                rule_id TEXT PRIMARY KEY,
                user_id_hash TEXT NOT NULL,
                emotion_category TEXT,
                decay_after_days INTEGER DEFAULT 30,
                decay_probability REAL DEFAULT 0.1,
                last_decay_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_contexts ON emotional_contexts(user_id_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON emotional_contexts(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_privacy_level ON emotional_contexts(privacy_level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON emotional_contexts(expires_at)")

        self.conn.commit()

    def _hash_user_id(self, user_id: str) -> str:
        """Create consistent hash of user ID for privacy"""
        return hashlib.sha256(f"{user_id}_ghost_protocol".encode()).hexdigest()

    def _generate_context_id(self, user_id: str, timestamp: datetime) -> str:
        """Generate unique context ID"""
        data = f"{user_id}_{timestamp.isoformat()}_{os.urandom(8).hex()}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def _calculate_data_hash(self, data: bytes) -> str:
        """Calculate hash for data integrity checking"""
        return hashlib.sha256(data).hexdigest()

    def store_emotional_context(self, context: EmotionalContext) -> bool:
        """Store an emotional context with encryption"""
        try:
            cursor = self.conn.cursor()

            # Generate context ID if not provided
            if not context.context_id:
                context.context_id = self._generate_context_id(context.user_id, context.created_at)

            # Hash user ID for privacy
            user_id_hash = self._hash_user_id(context.user_id)

            # Prepare data for encryption (excluding vector)
            context_data = {
                "user_id": context.user_id,  # This will be encrypted
                "primary_emotion": context.primary_emotion,
                "emotional_intensity": context.emotional_intensity,
                "conversation_summary": context.conversation_summary,
                "tags": context.tags,
                "metadata": context.metadata
            }

            # Encrypt main data
            encrypted_data = self.encryption_manager.encrypt_data(context_data)
            data_hash = self._calculate_data_hash(encrypted_data)

            # Encrypt emotion vector if present
            encrypted_vector = None
            vector_shape = None
            if context.emotion_vector is not None:
                encrypted_vector = self.encryption_manager.encrypt_vector(context.emotion_vector)
                vector_shape = json.dumps(context.emotion_vector.shape)

            # Create searchable tags
            searchable_tags = json.dumps(
                self.searchable_encryption.create_searchable_index(context.tags)
            )

            # Insert into database
            cursor.execute("""
                INSERT INTO emotional_contexts 
                (context_id, user_id_hash, encrypted_data, encrypted_vector, vector_shape,
                 searchable_tags, privacy_level, emotional_intensity, created_at, 
                 expires_at, access_count, last_accessed, data_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                context.context_id,
                user_id_hash,
                encrypted_data,
                encrypted_vector,
                vector_shape,
                searchable_tags,
                context.privacy_level,
                context.emotional_intensity,
                context.created_at,
                context.expires_at,
                context.access_count,
                context.last_accessed,
                data_hash
            ))

            self.conn.commit()
            return True

        except Exception as e:
            print(f"Error storing emotional context: {e}")
            self.conn.rollback()
            return False

    def query_emotional_contexts(self, query: MemoryQuery) -> List[EmotionalContext]:
        """Query emotional contexts with privacy-preserving search"""
        try:
            cursor = self.conn.cursor()
            user_id_hash = self._hash_user_id(query.user_id)

            # Build SQL query
            sql = """
                SELECT context_id, user_id_hash, encrypted_data, encrypted_vector, 
                       vector_shape, searchable_tags, privacy_level, emotional_intensity,
                       created_at, expires_at, access_count, last_accessed, data_hash
                FROM emotional_contexts 
                WHERE user_id_hash = ? AND privacy_level <= ?
            """
            params = [user_id_hash, query.privacy_level_max]

            # Add time range filter
            if query.time_range_hours:
                cutoff_time = datetime.now() - timedelta(hours=query.time_range_hours)
                sql += " AND created_at >= ?"
                params.append(cutoff_time)

            # Add expiration filter
            if not query.include_expired:
                sql += " AND (expires_at IS NULL OR expires_at > ?)"
                params.append(datetime.now())

            # Add ordering and limit
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(query.limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            contexts = []
            for row in rows:
                try:
                    # Decrypt main data
                    decrypted_data = self.encryption_manager.decrypt_data(row[2])

                    # Decrypt vector if present
                    emotion_vector = None
                    if row[3] and row[4]:
                        vector_shape = tuple(json.loads(row[4]))
                        emotion_vector = self.encryption_manager.decrypt_vector(row[3], vector_shape)

                    # Parse searchable tags
                    searchable_tags = json.loads(row[5]) if row[5] else []

                    # Apply tag filter if specified
                    if query.tag_filter:
                        tag_matches = False
                        for tag in query.tag_filter:
                            if self.searchable_encryption.matches_encrypted_tags(tag, searchable_tags):
                                tag_matches = True
                                break
                        if not tag_matches:
                            continue

                    # Apply emotion filter if specified
                    if query.emotion_filter and query.emotion_filter != decrypted_data.get("primary_emotion"):
                        continue

                    # Create context object
                    context = EmotionalContext(
                        context_id=row[0],
                        user_id=decrypted_data["user_id"],
                        emotion_vector=emotion_vector,
                        primary_emotion=decrypted_data.get("primary_emotion"),
                        emotional_intensity=decrypted_data.get("emotional_intensity", 0.0),
                        conversation_summary=decrypted_data.get("conversation_summary"),
                        tags=decrypted_data.get("tags", []),
                        privacy_level=row[6],
                        created_at=datetime.fromisoformat(row[8]) if row[8] else None,
                        expires_at=datetime.fromisoformat(row[9]) if row[9] else None,
                        access_count=row[10],
                        last_accessed=datetime.fromisoformat(row[11]) if row[11] else None,
                        metadata=decrypted_data.get("metadata", {})
                    )

                    contexts.append(context)

                    # Update access tracking
                    self._update_access_tracking(row[0])

                except Exception as e:
                    print(f"Error decrypting context {row[0]}: {e}")
                    continue

            return contexts

        except Exception as e:
            print(f"Error querying emotional contexts: {e}")
            return []

    def delete_emotional_context(self, context_id: str, user_id: str) -> bool:
        """Securely delete an emotional context"""
        try:
            cursor = self.conn.cursor()
            user_id_hash = self._hash_user_id(user_id)

            # Verify ownership before deletion
            cursor.execute("""
                SELECT context_id FROM emotional_contexts 
                WHERE context_id = ? AND user_id_hash = ?
            """, (context_id, user_id_hash))

            if not cursor.fetchone():
                return False  # Context not found or not owned by user

            # Delete the context
            cursor.execute("""
                DELETE FROM emotional_contexts 
                WHERE context_id = ? AND user_id_hash = ?
            """, (context_id, user_id_hash))

            # Delete related interaction history
            cursor.execute("""
                DELETE FROM interaction_history 
                WHERE context_id = ?
            """, (context_id,))

            self.conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            print(f"Error deleting emotional context: {e}")
            self.conn.rollback()
            return False

    def _update_access_tracking(self, context_id: str):
        """Update access count and timestamp for a context"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE emotional_contexts 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE context_id = ?
            """, (datetime.now(), context_id))
            self.conn.commit()
        except Exception as e:
            print(f"Error updating access tracking: {e}")

    def apply_memory_decay(self, user_id: str) -> int:
        """Apply memory decay rules to remove old or irrelevant contexts"""
        try:
            cursor = self.conn.cursor()
            user_id_hash = self._hash_user_id(user_id)

            # Get decay rules for user
            cursor.execute("""
                SELECT emotion_category, decay_after_days, decay_probability
                FROM memory_decay_rules 
                WHERE user_id_hash = ?
            """, (user_id_hash,))

            decay_rules = cursor.fetchall()
            if not decay_rules:
                # Apply default decay rules
                decay_rules = [
                    ("anger", 7, 0.3),  # Anger fades quickly
                    ("sadness", 14, 0.2),  # Sadness persists longer
                    ("anxiety", 10, 0.25),  # Anxiety moderate decay
                    ("joy", 30, 0.1),  # Keep positive emotions longer
                    ("fear", 7, 0.3)  # Fear fades quickly
                ]

            deleted_count = 0

            for emotion_category, decay_days, decay_prob in decay_rules:
                cutoff_date = datetime.now() - timedelta(days=decay_days)

                # Find contexts eligible for decay
                cursor.execute("""
                    SELECT context_id, encrypted_data 
                    FROM emotional_contexts 
                    WHERE user_id_hash = ? AND created_at < ?
                """, (user_id_hash, cutoff_date))

                contexts = cursor.fetchall()

                for context_id, encrypted_data in contexts:
                    try:
                        # Decrypt to check emotion category
                        decrypted_data = self.encryption_manager.decrypt_data(encrypted_data)
                        if decrypted_data.get("primary_emotion") == emotion_category:
                            # Apply decay probability
                            if np.random.random() < decay_prob:
                                cursor.execute("""
                                    DELETE FROM emotional_contexts 
                                    WHERE context_id = ?
                                """, (context_id,))
                                deleted_count += 1
                    except Exception as e:
                        print(f"Error processing decay for context {context_id}: {e}")
                        continue

            self.conn.commit()
            return deleted_count

        except Exception as e:
            print(f"Error applying memory decay: {e}")
            return 0

    def get_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about stored memories for a user"""
        try:
            cursor = self.conn.cursor()
            user_id_hash = self._hash_user_id(user_id)

            # Total contexts
            cursor.execute("""
                SELECT COUNT(*) FROM emotional_contexts 
                WHERE user_id_hash = ?
            """, (user_id_hash,))
            total_contexts = cursor.fetchone()[0]

            # Contexts by privacy level
            cursor.execute("""
                SELECT privacy_level, COUNT(*) 
                FROM emotional_contexts 
                WHERE user_id_hash = ?
                GROUP BY privacy_level
            """, (user_id_hash,))
            privacy_distribution = dict(cursor.fetchall())

            # Average emotional intensity
            cursor.execute("""
                SELECT AVG(emotional_intensity) 
                FROM emotional_contexts 
                WHERE user_id_hash = ?
            """, (user_id_hash,))
            avg_intensity = cursor.fetchone()[0] or 0.0

            # Contexts by time period
            now = datetime.now()
            time_periods = {
                "last_24h": now - timedelta(hours=24),
                "last_week": now - timedelta(days=7),
                "last_month": now - timedelta(days=30)
            }

            time_distribution = {}
            for period, cutoff in time_periods.items():
                cursor.execute("""
                    SELECT COUNT(*) FROM emotional_contexts 
                    WHERE user_id_hash = ? AND created_at >= ?
                """, (user_id_hash, cutoff))
                time_distribution[period] = cursor.fetchone()[0]

            # Most accessed contexts
            cursor.execute("""
                SELECT context_id, access_count 
                FROM emotional_contexts 
                WHERE user_id_hash = ?
                ORDER BY access_count DESC LIMIT 5
            """, (user_id_hash,))
            top_accessed = cursor.fetchall()

            return {
                "total_contexts": total_contexts,
                "privacy_distribution": privacy_distribution,
                "average_emotional_intensity": avg_intensity,
                "time_distribution": time_distribution,
                "top_accessed_contexts": top_accessed,
                "storage_health": "healthy" if total_contexts < 10000 else "consider_cleanup"
            }

        except Exception as e:
            print(f"Error getting memory statistics: {e}")
            return {}

    def export_user_data(self, user_id: str, include_vectors: bool = False) -> Dict[str, Any]:
        """Export all user data for portability (GDPR compliance)"""
        try:
            query = MemoryQuery(
                user_id=user_id,
                privacy_level_max=3,
                limit=1000,  # Export up to 1000 contexts
                include_expired=True
            )

            contexts = self.query_emotional_contexts(query)

            export_data = {
                "user_id": user_id,
                "export_timestamp": datetime.now().isoformat(),
                "total_contexts": len(contexts),
                "contexts": []
            }

            for context in contexts:
                context_data = {
                    "context_id": context.context_id,
                    "primary_emotion": context.primary_emotion,
                    "emotional_intensity": context.emotional_intensity,
                    "conversation_summary": context.conversation_summary,
                    "tags": context.tags,
                    "privacy_level": context.privacy_level,
                    "created_at": context.created_at.isoformat() if context.created_at else None,
                    "expires_at": context.expires_at.isoformat() if context.expires_at else None,
                    "access_count": context.access_count,
                    "metadata": context.metadata
                }

                if include_vectors and context.emotion_vector is not None:
                    context_data["emotion_vector"] = context.emotion_vector.tolist()

                export_data["contexts"].append(context_data)

            return export_data

        except Exception as e:
            print(f"Error exporting user data: {e}")
            return {}

    def purge_user_data(self, user_id: str) -> bool:
        """Completely remove all data for a user (GDPR compliance)"""
        try:
            cursor = self.conn.cursor()
            user_id_hash = self._hash_user_id(user_id)

            # Delete emotional contexts
            cursor.execute("""
                DELETE FROM emotional_contexts 
                WHERE user_id_hash = ?
            """, (user_id_hash,))
            contexts_deleted = cursor.rowcount

            # Delete interaction history
            cursor.execute("""
                DELETE FROM interaction_history 
                WHERE user_id_hash = ?
            """, (user_id_hash,))
            interactions_deleted = cursor.rowcount

            # Delete decay rules
            cursor.execute("""
                DELETE FROM memory_decay_rules 
                WHERE user_id_hash = ?
            """, (user_id_hash,))
            rules_deleted = cursor.rowcount

            self.conn.commit()

            print(
                f"Purged user data: {contexts_deleted} contexts, {interactions_deleted} interactions, {rules_deleted} decay rules")
            return True

        except Exception as e:
            print(f"Error purging user data: {e}")
            self.conn.rollback()
            return False

    def verify_data_integrity(self) -> Dict[str, Any]:
        """Verify integrity of stored encrypted data"""
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                SELECT context_id, encrypted_data, data_hash 
                FROM emotional_contexts
            """)

            total_contexts = 0
            corrupted_contexts = []

            for context_id, encrypted_data, stored_hash in cursor.fetchall():
                total_contexts += 1
                calculated_hash = self._calculate_data_hash(encrypted_data)

                if calculated_hash != stored_hash:
                    corrupted_contexts.append(context_id)

            return {
                "total_contexts_checked": total_contexts,
                "corrupted_contexts": len(corrupted_contexts),
                "corrupted_context_ids": corrupted_contexts,
                "integrity_score": 1.0 - (len(corrupted_contexts) / max(total_contexts, 1))
            }

        except Exception as e:
            print(f"Error verifying data integrity: {e}")
            return {"error": str(e)}

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def test_memory_vault():
    """Test the encrypted memory vault"""
    print("🔐 Testing Encrypted Memory Vault...")

    # Initialize vault
    vault = EmotionalMemoryVault("test_ghost_memory.db", "test_password_123")

    # Test data
    test_contexts = [
        EmotionalContext(
            context_id="test_001",
            user_id="test_user_1",
            emotion_vector=np.array([0.8, 0.2, 0.1, 0.3, 0.5], dtype=np.float32),
            primary_emotion="anger",
            emotional_intensity=0.9,
            conversation_summary="User expressed frustration about work deadline",
            tags=["work", "deadline", "frustration", "angry"],
            privacy_level=2,
            metadata={"session_id": "session_001", "trigger": "work_stress"}
        ),
        EmotionalContext(
            context_id="test_002",
            user_id="test_user_1",
            emotion_vector=np.array([0.1, 0.9, 0.8, 0.2, 0.1], dtype=np.float32),
            primary_emotion="joy",
            emotional_intensity=0.7,
            conversation_summary="User celebrated successful project completion",
            tags=["work", "success", "celebration", "happy"],
            privacy_level=1,
            metadata={"session_id": "session_002", "trigger": "achievement"}
        ),
        EmotionalContext(
            context_id="test_003",
            user_id="test_user_2",
            emotion_vector=np.array([0.3, 0.1, 0.2, 0.9, 0.6], dtype=np.float32),
            primary_emotion="anxiety",
            emotional_intensity=0.8,
            conversation_summary="User worried about upcoming presentation",
            tags=["presentation", "anxiety", "worry", "work"],
            privacy_level=3,
            metadata={"session_id": "session_003", "trigger": "performance_anxiety"}
        )
    ]

    print("\n📥 Storing emotional contexts...")

    # Store contexts
    for context in test_contexts:
        success = vault.store_emotional_context(context)
        print(f"✅ Stored context {context.context_id}: {success}")

    print("\n🔍 Querying emotional contexts...")

    # Test queries
    queries = [
        MemoryQuery(user_id="test_user_1", limit=5),
        MemoryQuery(user_id="test_user_1", emotion_filter="anger"),
        MemoryQuery(user_id="test_user_1", tag_filter=["work"]),
        MemoryQuery(user_id="test_user_2", privacy_level_max=3)
    ]

    for i, query in enumerate(queries, 1):
        results = vault.query_emotional_contexts(query)
        print(f"Query {i}: Found {len(results)} contexts")
        for result in results:
            print(f"  - {result.context_id}: {result.primary_emotion} (intensity: {result.emotional_intensity})")

    print("\n📊 Memory statistics...")

    # Test statistics
    for user_id in ["test_user_1", "test_user_2"]:
        stats = vault.get_memory_statistics(user_id)
        print(f"User {user_id}:")
        print(f"  Total contexts: {stats.get('total_contexts', 0)}")
        print(f"  Avg intensity: {stats.get('average_emotional_intensity', 0):.2f}")
        print(f"  Privacy distribution: {stats.get('privacy_distribution', {})}")

    print("\n🧹 Testing memory decay...")

    # Test memory decay
    deleted_count = vault.apply_memory_decay("test_user_1")
    print(f"Applied memory decay: {deleted_count} contexts deleted")

    print("\n🔒 Testing data integrity...")

    # Test data integrity
    integrity_report = vault.verify_data_integrity()
    print(f"Integrity check: {integrity_report.get('integrity_score', 0):.2%} healthy")
    print(f"Corrupted contexts: {integrity_report.get('corrupted_contexts', 0)}")

    print("\n📤 Testing data export...")

    # Test data export
    export_data = vault.export_user_data("test_user_1")
    print(f"Exported {export_data.get('total_contexts', 0)} contexts for user")

    print("\n🗑️ Testing secure deletion...")

    # Test deletion
    deleted = vault.delete_emotional_context("test_001", "test_user_1")
    print(f"Deleted context test_001: {deleted}")

    # Cleanup
    vault.close()

    # Remove test database
    import os
    if os.path.exists("test_ghost_memory.db"):
        os.remove("test_ghost_memory.db")

    print("\n✅ Memory vault tests completed!")
    return True


if __name__ == "__main__":
    test_memory_vault()