class AlertService:
    def check_alerts(self, repo, person_id: int) -> list[dict]:
        """인식 시점에 호출. recognition 라우터에서 매칭 결과마다 호출."""
        rules = repo.alert_rule.get_active_rules(person_id)
        return [
            {
                "alert_type": r.alert_type
                if hasattr(r, "alert_type")
                else r.get("alert_type"),
                "message": r.message if hasattr(r, "message") else r.get("message"),
            }
            for r in rules
        ]


alert_service = AlertService()
