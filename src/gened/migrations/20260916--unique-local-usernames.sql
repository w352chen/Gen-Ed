-- SPDX-FileCopyrightText: 2026 Mark Liffiton <liffiton@gmail.com>
--
-- SPDX-License-Identifier: AGPL-3.0-only

CREATE UNIQUE INDEX IF NOT EXISTS auth_local_username_nocase_idx
ON auth_local(username COLLATE NOCASE);
